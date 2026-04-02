import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
load_dotenv()


def _available_memory_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None

    available_kib = None
    with meminfo.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                available_kib = int(line.split()[1])
                break

    if available_kib is None:
        return None

    return available_kib / (1024 ** 2)

class Retrieval:
    """
    This is a class to handle the retrieval side of the pipeline. 
    Here prefixes are embedded and queries are handled.
    """
    
    def __init__(
                self, 
                model_id: str = None, 
                dataset: str = None,
            ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = QdrantClient(host="localhost",
                                    grpc_port=int(os.getenv("QDRANT_GRPC_PORT")),
                                    prefer_grpc=True,
                                )
        
        embedding_models = {
            "minilm": "sentence-transformers/all-MiniLM-L12-v2",
            "snowflake": "Snowflake/snowflake-arctic-embed-m-v2.0",
        }

        if model_id not in embedding_models:
            raise ValueError(f"Unsupported model_id: {model_id}")

        self.model_id = model_id
        self.dataset = Path(dataset)
        if model_id == "minilm":
            self.model = SentenceTransformer(embedding_models[model_id])
            self.tokenizer = None
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            available_memory_gib = _available_memory_gib()
            if self.device.type == "cpu" and available_memory_gib is not None and available_memory_gib < 2.5:
                raise RuntimeError(
                    "The Snowflake embedding model needs more available RAM than this machine currently has. "
                    f"Detected about {available_memory_gib:.2f} GiB available on CPU. "
                    "Use model_id='minilm', free up memory, or run on a GPU-enabled machine."
                )

            self.tokenizer = AutoTokenizer.from_pretrained(embedding_models[model_id])
            self.model = AutoModel.from_pretrained(
                embedding_models[model_id],
                add_pooling_layer=False,
                trust_remote_code=True,
                attn_implementation="eager",
                use_memory_efficient_attention=False,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            self.model.eval()
            self.dimension = self.model.config.hidden_size
            
    @property
    def collection_name(self):
        return f"{self.model_id}_{self.dataset.stem}"

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
    
    def encode(self, texts: list[str], is_query: bool = False):
        if self.model_id == "minilm":
            return self.model.encode(texts, normalize_embeddings=True)

        if is_query:
            texts = [f"query: {text}" for text in texts]

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}

        with torch.no_grad():
            embeddings = self.model(**tokens)[0][:, 0]

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()
    
    def store_embeddings(self, batch_size: int = 1):
        
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
        
        qvec = self.encode([query_full_prefix], is_query=True)[0].tolist()

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            limit=top_k,
            with_payload=True,
        )
        hits = res.points

        context = {}
        for rank, h in enumerate(hits, start=1):
            p = h.payload or {}
        
            context[f"trace_{rank}"] = {
                "prefix": p.get("prefix", ""),
                "prediction": p.get("prediction", ""),
                "score": round(float(h.score), 4),
            }

        return context, hits
