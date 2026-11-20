"""
Idempotent indexing into Pinecone.

    python -m src.ingestion.indexer            # sync every file in Data/documents
    python -m src.ingestion.indexer --rebuild  # wipe the namespace first

Re-running the indexer never creates duplicates:
  * chunk IDs are deterministic (see chunker.make_chunk_id);
  * a file whose SHA-256 matches the manifest is skipped;
  * vectors of a changed file that no longer exist are deleted.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from config import (
    EMBED_BATCH_SIZE,
    MANIFEST_PATH,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_DIMENSION,
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
)
from src.embeddings.ollama_embeddings import get_embeddings
from src.ingestion.chunker import chunk_documents, file_slug
from src.ingestion.loader import list_documents, load_document
from src.utils.logger import logger

Progress = Optional[Callable[[float, str], None]]


# ------------------------------------------------------------------
# Pinecone
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_index():
    """Return the Pinecone index, creating it on first use."""

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing. Add it to your .env file.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        logger.info(f"Creating Pinecone index {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    return pc.Index(PINECONE_INDEX_NAME)


def existing_ids(file_name: str) -> List[str]:
    """IDs already stored for one file (IDs are prefixed with the file slug)."""

    prefix = file_slug(file_name) + "#"
    ids: List[str] = []
    for page in get_index().list(prefix=prefix, namespace=PINECONE_NAMESPACE):
        ids.extend(page)
    return ids


def vector_count() -> int:
    stats = get_index().describe_index_stats()
    namespace = stats.namespaces.get(PINECONE_NAMESPACE)
    return int(namespace.vector_count) if namespace else 0


# ------------------------------------------------------------------
# Manifest (what has been indexed, so unchanged files are skipped)
# ------------------------------------------------------------------

def load_manifest() -> Dict[str, dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: Dict[str, dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------
# Indexing
# ------------------------------------------------------------------

def index_file(path: Path, force: bool = False, progress: Progress = None) -> dict:
    """Chunk, embed and upsert one file. Returns a summary dict."""

    path = Path(path)
    manifest = load_manifest()
    digest = file_sha256(path)

    if not force and manifest.get(path.name, {}).get("sha256") == digest:
        logger.info(f"{path.name} unchanged, skipping")
        return {**manifest[path.name], "file_name": path.name, "status": "unchanged"}

    start = time.perf_counter()
    pages = load_document(path)
    chunks = chunk_documents(pages)
    if not chunks:
        raise ValueError(f"No indexable text found in {path.name}")

    index = get_index()
    embeddings = get_embeddings()

    for begin in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[begin:begin + EMBED_BATCH_SIZE]
        vectors = embeddings.embed_documents([c.page_content for c in batch])

        if len(vectors[0]) != PINECONE_DIMENSION:
            raise ValueError(
                f"Embedding dimension {len(vectors[0])} does not match "
                f"PINECONE_DIMENSION={PINECONE_DIMENSION}"
            )

        index.upsert(
            namespace=PINECONE_NAMESPACE,
            vectors=[
                {
                    "id": c.metadata["chunk_id"],
                    "values": v,
                    "metadata": {
                        "text": c.page_content,
                        "file_name": c.metadata["file_name"],
                        "page": int(c.metadata["page"]),
                        "chunk_index": int(c.metadata["chunk_index"]),
                    },
                }
                for c, v in zip(batch, vectors)
            ],
        )

        done = min(begin + EMBED_BATCH_SIZE, len(chunks))
        if progress:
            progress(done / len(chunks), f"{path.name}: {done}/{len(chunks)} chunks")

    # Remove vectors from a previous version of this file.
    new_ids = {c.metadata["chunk_id"] for c in chunks}
    stale = [i for i in existing_ids(path.name) if i not in new_ids]
    for begin in range(0, len(stale), 1000):
        index.delete(ids=stale[begin:begin + 1000], namespace=PINECONE_NAMESPACE)

    summary = {
        "sha256": digest,
        "pages": len(pages),
        "chunks": len(chunks),
        "stale_removed": len(stale),
        "indexing_seconds": round(time.perf_counter() - start, 2),
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest[path.name] = summary
    save_manifest(manifest)

    logger.info(f"Indexed {path.name}: {summary}")
    return {**summary, "file_name": path.name, "status": "indexed"}


def remove_file(file_name: str) -> int:
    """Delete one document's vectors and manifest entry."""

    ids = existing_ids(file_name)
    for begin in range(0, len(ids), 1000):
        get_index().delete(ids=ids[begin:begin + 1000], namespace=PINECONE_NAMESPACE)

    manifest = load_manifest()
    manifest.pop(file_name, None)
    save_manifest(manifest)
    return len(ids)


def sync_all(rebuild: bool = False, progress: Progress = None) -> List[dict]:
    """Index every document in DOCUMENTS_DIR."""

    if rebuild:
        logger.info(f"Clearing namespace '{PINECONE_NAMESPACE}'")
        try:
            get_index().delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        except Exception as error:  # namespace may not exist yet
            logger.info(f"Nothing to clear: {error}")
        save_manifest({})

    files = list_documents()
    if not files:
        raise FileNotFoundError("No supported documents found in Data/documents/")

    return [index_file(f, force=rebuild, progress=progress) for f in files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents into Pinecone")
    parser.add_argument("--rebuild", action="store_true", help="wipe the namespace and re-index everything")
    args = parser.parse_args()

    for result in sync_all(rebuild=args.rebuild, progress=lambda _, msg: print(f"  {msg}")):
        print(result)
    print("Vectors in namespace:", vector_count())
