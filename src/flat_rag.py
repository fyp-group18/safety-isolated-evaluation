import json
import logging
from pathlib import Path

import chromadb
from tqdm import tqdm

from src.config import (
    CHUNKS_PATH,
    MODEL_EMBEDDING,
    TOP_K,
    VECTOR_STORE_DIR,
)
from src.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_EMBEDDING)
        logger.info(f"Loaded embedding model: {MODEL_EMBEDDING}")
    return _model


def _embed_single(text: str) -> list[float]:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def _embed_batch(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def _load_chunks() -> list[dict]:
    with open(CHUNKS_PATH) as f:
        return json.load(f)


def build_index(batch_size: int = 256) -> chromadb.Collection:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    existing = client.list_collections()
    existing_names = [c.name if hasattr(c, 'name') else c for c in existing]
    if "flat_rag" in existing_names:
        coll = client.get_collection("flat_rag")
        count = coll.count()
        if count > 0:
            logger.info(f"Index already exists with {count} chunks, skipping rebuild")
            return coll

    chunks = _load_chunks()
    logger.info(f"Building index for {len(chunks)} chunks using {MODEL_EMBEDDING}")

    try:
        client.delete_collection("flat_rag")
    except Exception:
        pass

    coll = client.create_collection(
        name="flat_rag",
        metadata={"hnsw:space": "cosine"},
    )

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "document_id": str(c.get("document_id", "")),
                "document_name": c.get("document_name", ""),
                "section_header": c.get("section_header") or "",
                "section_code": c.get("section_code") or "",
                "page_start": c.get("page_start") or 0,
                "sequence_index": c.get("sequence_index") or 0,
                "has_safety_content": c.get("has_safety_content", False),
            }
            for c in batch
        ]

        embeddings = _embed_batch(texts)

        coll.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    logger.info(f"Index built: {coll.count()} chunks indexed")
    return coll


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return client.get_collection("flat_rag")


def retrieve(
    query: str,
    k: int = TOP_K,
    collection: chromadb.Collection | None = None,
    where: dict | None = None,
) -> list[RetrievedChunk]:
    if collection is None:
        collection = get_collection()

    query_embedding = _embed_single(query)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    retrieved = []
    for i in range(len(results["ids"][0])):
        cosine_distance = results["distances"][0][i]
        cosine_score = 1.0 - cosine_distance

        retrieved.append(
            RetrievedChunk(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                cosine_score=cosine_score,
            )
        )

    return retrieved


def retrieve_safety(
    equipment_type: str,
    k: int = 10,
    collection: chromadb.Collection | None = None,
) -> list[RetrievedChunk]:
    query = f"safety warnings hazards precautions {equipment_type}"
    return retrieve(
        query=query,
        k=k,
        collection=collection,
        where={"has_safety_content": True},
    )
