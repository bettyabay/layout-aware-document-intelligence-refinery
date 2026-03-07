"""Vector store implementation using ChromaDB."""

from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from sentence_transformers import SentenceTransformer


class VectorStore:
    """ChromaDB vector store for semantic search."""

    def __init__(self, persist_dir: Path | str = ".refinery/chroma_db"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "Document Intelligence Refinery chunks"}
        )
        
        # Initialize embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def ingest(self, doc_id: str, ldus: list[dict]) -> None:
        """Ingest LDUs into vector store."""
        texts = [ldu.get("text", "") for ldu in ldus]
        ids = [ldu.get("id", f"{doc_id}-{i}") for i, ldu in enumerate(ldus)]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts).tolist()
        
        # Prepare metadata
        metadatas = []
        for ldu in ldus:
            metadatas.append({
                "doc_id": doc_id,
                "chunk_type": ldu.get("chunk_type", "paragraph"),
                "page_refs": str(ldu.get("page_refs", [])),
                "parent_section": ldu.get("parent_section", ""),
            })
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas,
        )

    def search(self, query: str, doc_ids: Optional[list[str]] = None, k: int = 5) -> list[dict]:
        """Semantic search."""
        # Generate query embedding
        query_embedding = self.embedder.encode([query]).tolist()[0]
        
        # Build where clause
        where = {}
        if doc_ids:
            where["doc_id"] = {"$in": doc_ids}
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where if where else None,
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                })
        
        return formatted_results

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks for a document."""
        # ChromaDB doesn't have direct delete by metadata, so we query first
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
