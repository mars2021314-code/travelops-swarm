import hashlib
import math
import uuid
from typing import Any

from langchain_openai import OpenAIEmbeddings
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.utils import get_qdrant_client
from customer_support_chat.app.services.vectordb.chunkenizer import recursive_character_splitting
from customer_support_chat.app.core.logger import logger

VECTOR_SIZE = 1536
settings = get_settings()
_embedding_client = None


class VectorDB:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.client = get_qdrant_client()
        self.create_collection()

    def create_collection(self):
        try:
            self.client.get_collection(collection_name=self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except (qdrant_exceptions.UnexpectedResponse, ValueError) as exc:
            if isinstance(exc, qdrant_exceptions.UnexpectedResponse) and exc.status_code != 404:
                raise
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def _get_embedding_client(self):
        global _embedding_client
        if _embedding_client is not None:
            return _embedding_client

        if not settings.OPENAI_API_KEY:
            return None

        _embedding_client = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
        return _embedding_client

    def _generate_hash_embedding(self, content: str) -> list[float]:
        text = content or ""
        vector = [0.0] * VECTOR_SIZE
        tokens = text.lower().split()
        if not tokens:
            tokens = [text]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % VECTOR_SIZE
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def generate_embedding(self, content):
        text = content or ""
        embedding_client = self._get_embedding_client()
        if embedding_client is None:
            return self._generate_hash_embedding(text)

        try:
            return embedding_client.embed_query(text)
        except Exception as exc:
            logger.warning(
                "Embedding provider failed for collection %s. Falling back to hash embeddings. Error: %s",
                self.collection_name,
                str(exc),
            )
            return self._generate_hash_embedding(text)

    def upsert_vector(self, doc_id, chunk_text, embedding, url, chunk_index):
        chunk_id = str(uuid.uuid4())
        payload = {
            "url": url,
            "document_id": str(doc_id),
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=chunk_id, vector=embedding, payload=payload)
            ]
        )

    def upsert_payload(
        self,
        *,
        point_id: str,
        content: str,
        payload: dict[str, Any],
    ) -> None:
        embedding = self.generate_embedding(content)
        merged_payload = dict(payload)
        merged_payload["chunk_text"] = content
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=merged_payload,
                )
            ],
        )

    def create_embeddings(self, docs):
        for doc_id, content, url in docs:
            if content is None:
                logger.warning(f"Skipping doc_id {doc_id} because content is None")
                continue

            chunks = recursive_character_splitting(content)
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Generating embedding for doc_id: {doc_id}, chunk: {i+1}")
                    embedding = self.generate_embedding(chunk)
                    self.upsert_vector(doc_id, chunk, embedding, url, i)
                except Exception as e:
                    logger.error(f"Failed to generate or store embedding for doc_id: {doc_id}, chunk: {i+1}, error: {str(e)}")

        logger.info("Completed generating embeddings for all documents")

    def search(self, query, k=3, filters: dict[str, Any] | None = None):
        query_embedding = self.generate_embedding(query)
        conditions = []
        for key, value in (filters or {}).items():
            if value is None:
                continue
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        query_filter = Filter(must=conditions) if conditions else None
        if hasattr(self.client, "search"):
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,
                query_filter=query_filter,
            )
            return search_result

        query_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            with_payload=True,
            query_filter=query_filter,
        )
        search_result = getattr(query_result, "points", query_result)
        return search_result
