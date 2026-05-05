import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from transformers import AutoModelForSequenceClassification, AutoTokenizer
load_dotenv()

class Retrieval:
    """
    This is a class to handle the retrieval side of the pipeline. 
    Here prefixes are embedded and queries are handled.
    """
    
    def __init__(
                self, 
                model_id: str = None, 
                dataset: str = None,
                test_set: str = None,
                reranker_model_id: str | None = None,
                rerank_pool_k: int | None = None,
            ):
        self.client = QdrantClient(host="localhost",
                                   grpc_port=int(os.getenv("QDRANT_GRPC_PORT")),
                                   prefer_grpc=True,
                                   timeout=60,
                                )
        
        embedding_models = {
            "minilm": "sentence-transformers/all-MiniLM-L12-v2",
            "bge": "BAAI/bge-small-en-v1.5",
        }
        reranker_models = {
            "bge": "BAAI/bge-reranker-base",
        }

        if model_id not in embedding_models:
            raise ValueError(f"Unsupported model_id: {model_id}")
        if reranker_model_id is not None and reranker_model_id not in reranker_models:
            raise ValueError(f"Unsupported reranker_model_id: {reranker_model_id}")
        if rerank_pool_k is not None and rerank_pool_k <= 0:
            raise ValueError("rerank_pool_k must be > 0")

        self.model_id = model_id
        self.dataset = Path(dataset)
        self.test_set = Path(test_set) if test_set is not None else self.dataset.with_name("test.csv")
        self.model = SentenceTransformer(embedding_models[model_id])
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.reranker_model_id = reranker_model_id
        self.rerank_pool_k = rerank_pool_k
        self.reranker_model_name = reranker_models.get(reranker_model_id)
        self._reranker_tokenizer = None
        self._reranker_model = None
        self._reranker_device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def variant_name(self) -> str:
        if self.dataset.name in {"retrieval.csv", "test.csv"}:
            return self.dataset.parent.name
        return self.dataset.stem
            
    @property
    def collection_name(self):
        return f"{self.model_id}_{self.variant_name}"

    def initialize_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
        return self.collection_name
    
    def encode(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True)
    
    def store_embeddings(self, batch_size: int = 32):
        
        self.initialize_collection()
        
        csv_prefixes = self.dataset

        df = pd.read_csv(csv_prefixes)
        prefixes = df["prefix"].tolist()
        predictions = df["prediction"].tolist()
        
        for start in range(0, len(prefixes), batch_size):
            end = start + batch_size
            batch_prefixes = prefixes[start:end]
            batch_predictions = predictions[start:end] 
            embeddings = self.encode(batch_prefixes)

            points = [
                models.PointStruct(
                    id=start + i,
                    vector=embeddings[i].tolist(),
                    payload={
                        "prefix": batch_prefixes[i], 
                        "prediction": batch_predictions[i]
                    },
                )
                for i in range(len(batch_prefixes))
            ]

            self.client.upsert(
                collection_name=self.collection_name, 
                points=points
                )
        return
    
    def retrieve_similar_prefixes(
        self, 
        query_full_prefix: str, 
        top_k: int = 5
    ) -> tuple[dict, list]:
        qvec = self.encode([query_full_prefix])[0].tolist()
        retrieval_limit = top_k
        if self.reranker_model_id and self.rerank_pool_k is not None:
            retrieval_limit = max(top_k, self.rerank_pool_k)

        res = None
        for attempt in range(4):
            try:
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=qvec,
                    limit=retrieval_limit,
                    with_payload=True,
                    timeout=60,
                )
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                print(
                    f"Qdrant query failed (attempt {attempt + 1}/4): {exc}. Retrying in 10 seconds..."
                )
                time.sleep(10)

        hits = list(res.points)
        rerank_scores: dict[int, float] = {}
        if self.reranker_model_id and len(hits) > top_k:
            hits, rerank_scores = self._rerank_hits(query_full_prefix, hits, top_k)
        else:
            hits = hits[:top_k]

        context = {}
        for rank, h in enumerate(hits, start=1):
            p = h.payload or {}
            score = rerank_scores.get(int(h.id), float(h.score))
        
            context[f"trace_{rank}"] = {
                "prefix": p.get("prefix", ""),
                "prediction": p.get("prediction", ""),
                "score": round(float(score), 4),
            }

        return context, hits

    def _ensure_reranker_loaded(self) -> None:
        if self._reranker_model is not None and self._reranker_tokenizer is not None:
            return
        if not self.reranker_model_name:
            raise ValueError("Reranker requested without a valid reranker model name.")

        self._reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self._reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name)
        self._reranker_model = self._reranker_model.to(self._reranker_device)
        self._reranker_model.eval()

    def _rerank_hits(
        self,
        query_full_prefix: str,
        hits: list,
        top_k: int,
    ) -> tuple[list, dict[int, float]]:
        self._ensure_reranker_loaded()

        pairs = []
        hit_ids = []
        for hit in hits:
            payload = hit.payload or {}
            candidate_prefix = payload.get("prefix", "")
            pairs.append((query_full_prefix, candidate_prefix))
            hit_ids.append(int(hit.id))

        with torch.no_grad():
            inputs = self._reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._reranker_device) for k, v in inputs.items()}
            outputs = self._reranker_model(**inputs)
            logits = outputs.logits.squeeze(-1)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            scores = logits.detach().cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(zip(scores, hits, hit_ids), key=lambda x: x[0], reverse=True)
        top_ranked = ranked[:top_k]

        reranked_hits = [item[1] for item in top_ranked]
        rerank_scores = {int(item[2]): float(item[0]) for item in top_ranked}
        return reranked_hits, rerank_scores
