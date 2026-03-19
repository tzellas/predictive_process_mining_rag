import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
load_dotenv()

EMBEDDING_MODELS = ["sentence-transformers/all-MiniLM-L12-v2"]
class RAGConfig:
    def __init__(
                self, 
                model_id: str = None, 
                dataset: str = None,
            ):
        self.client = QdrantClient(host="localhost",
                                    grpc_port=int(os.getenv("QDRANT_GRPC_PORT")),
                                    prefer_grpc=True,
                                )
             
        self.model_id = model_id
        self.dataset = Path(dataset)
        self.model = self.resolve_model()
        self.dimension = self.get_dimension()
            
    @property
    def collection_name(self):
        return f"{self.model_id.replace('/', '_')}_{self.dataset.stem}"

    def resolve_model(self):
        if self.model_id == "sentence-transformers/all-MiniLM-L12-v2":
            return SentenceTransformer(self.model_id)
        raise ValueError(f"Unsupported model_id: {self.model_id}")

    def get_dimension(self):
        if self.model_id == "sentence-transformers/all-MiniLM-L12-v2":
            return self.model.get_sentence_embedding_dimension()

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
        if self.model_id.startswith("sentence-transformers"):
            return self.model.encode(texts, normalize_embeddings=True)
        raise ValueError(f"Unsupported encode provider for model_id: {self.model_id}")
    
    def store_embeddings(self, batch_size: int = 256):
        
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