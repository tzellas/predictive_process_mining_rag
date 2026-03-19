import pandas as pd
from pathlib import Path
from rag_config import RAGConfig
from qdrant_client import models
from qdrant_client.http.models import Distance, VectorParams

def store_embeddings(
    config: RAGConfig,
) -> None:
    
    csv_prefixes = config.dataset

    df = pd.read_csv(csv_prefixes)
    prefixes = df["prefix"].tolist()
    predictions = df["prediction"].tolist()
    
    embeddings = config.encode(prefixes)

    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"prefix": prefixes[i], "prediction": predictions[i]},
        )
        for i in range(len(prefixes))
    ]

    config.client.upsert(
        collection_name=config.collection_name, 
        points=points
        )
    return 


def retrieve_similar_prefixes(
    config: RAGConfig, 
    query_full_prefix: str, 
    top_k: int = 5
) -> tuple[dict, list]:
    
    qvec = config.encode([query_full_prefix])[0].tolist()

    res = config.client.query_points(
        collection_name=config.collection_name,
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
