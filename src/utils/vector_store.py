"""Vector store manager for LDUs.

This module provides a ChromaDB-based vector store for storing and retrieving
Logical Document Units (LDUs) for semantic search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.models.ldu import LDU

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for LDUs.

    Stores LDUs with their embeddings and metadata for semantic search.
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "ldus",
    ):
        """Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data.
            embedding_model: Sentence transformer model name for embeddings.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

    def add_ldus(
        self,
        doc_id: str,
        doc_name: str,
        ldus: List[LDU],
        batch_size: int = 100,
    ) -> None:
        """Add LDUs to the vector store.

        Args:
            doc_id: Document identifier.
            doc_name: Document name.
            ldus: List of LDUs to add.
            batch_size: Batch size for adding documents.
        """
        if not ldus:
            logger.warning(f"No LDUs to add for document {doc_id}")
            return

        logger.info(f"Adding {len(ldus)} LDUs to vector store for document {doc_id}")

        # Prepare data for ChromaDB
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []
        embeddings: List[List[float]] = []

        for ldu in ldus:
            # Use content_hash as unique ID
            chunk_id = f"{doc_id}_{ldu.content_hash}"
            ids.append(chunk_id)

            # Store content as document
            documents.append(ldu.content)

            # Store metadata
            metadata = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunk_type": ldu.chunk_type,
                "page_refs": json.dumps(ldu.page_refs),
                "content_hash": ldu.content_hash,
                "parent_section": ldu.parent_section or "",
                "token_count": ldu.token_count,
            }
            # Add bounding box if available
            if ldu.bounding_box:
                metadata["bbox"] = json.dumps(ldu.bounding_box)

            metadatas.append(metadata)

            # Generate embedding
            embedding = self.embedding_model.encode(ldu.content).tolist()
            embeddings.append(embedding)

        # Add in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_embeds = embeddings[i : i + batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embeds,
            )

        logger.info(f"Successfully added {len(ldus)} LDUs to vector store")

    def search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Tuple[LDU, float]]:
        """Search for similar LDUs.

        Args:
            query: Search query text.
            doc_id: Optional document ID to filter results.
            top_k: Number of results to return.
            filter_metadata: Additional metadata filters.

        Returns:
            List of (LDU, similarity_score) tuples, sorted by relevance.
        """
        # Build where clause for filtering
        where = {}
        if doc_id:
            where["doc_id"] = doc_id
        if filter_metadata:
            where.update(filter_metadata)

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
        )

        # Convert results to LDU objects
        ldus_with_scores: List[Tuple[LDU, float]] = []

        if results["ids"] and len(results["ids"][0]) > 0:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert distance to similarity (cosine distance -> similarity)
                similarity = 1.0 - distance

                # Reconstruct LDU from metadata and document
                ldu = LDU(
                    content=document,
                    chunk_type=metadata.get("chunk_type", "paragraph"),
                    page_refs=json.loads(metadata.get("page_refs", "[1]")),
                    bounding_box=json.loads(metadata.get("bbox", "{}")),
                    parent_section=metadata.get("parent_section") or None,
                    token_count=metadata.get("token_count", 0),
                    content_hash=metadata.get("content_hash", ""),
                )

                ldus_with_scores.append((ldu, similarity))

        return ldus_with_scores

    def delete_document(self, doc_id: str) -> None:
        """Delete all LDUs for a document.

        Args:
            doc_id: Document identifier.
        """
        # ChromaDB doesn't support bulk delete by metadata, so we need to query first
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} LDUs for document {doc_id}")
        else:
            logger.warning(f"No LDUs found for document {doc_id}")

    def get_document_count(self, doc_id: Optional[str] = None) -> int:
        """Get the number of LDUs stored.

        Args:
            doc_id: Optional document ID to filter.

        Returns:
            Number of LDUs.
        """
        if doc_id:
            results = self.collection.get(where={"doc_id": doc_id})
            return len(results["ids"]) if results["ids"] else 0
        return self.collection.count()


__all__ = ["VectorStore"]
