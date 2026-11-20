"""
Cleaning and chunking.

Improvements over the first version:
  * Unicode NFKC normalisation, so PDF ligatures ("ﬁ") become plain "fi"
    and the text matches what users type.
  * Line-break hyphenation is repaired ("algo-\\nrithm" -> "algorithm").
  * Running headers/footers are detected statistically (a line repeated on
    many pages) instead of hard-coding one book's title.
  * Chunk size and overlap come from config.
  * Every chunk gets a deterministic ID, which makes re-indexing idempotent.
"""

import hashlib
import re
import unicodedata
from collections import Counter
from typing import AbstractSet, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE

MIN_PAGE_CHARS = 80      # pages shorter than this after cleaning are covers/blank
MIN_CHUNK_CHARS = 150    # fragments shorter than this carry no useful context
BOILERPLATE_RATIO = 0.2  # a line on >=20% of pages is a header/footer


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Some LaTeX PDFs lose the ff/ffi ligature glyph and pypdf emits NUL:
    # "di\x00erent" -> "different", "e\x00cient" -> "efficient".
    text = re.sub(r"\x00(?=c)", "ffi", text).replace("\x00", "ff")
    # Repair words split across lines: "hyphen-\nated" -> "hyphenated"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text


def _line_key(line: str) -> str:
    """Compare lines with digits masked, so 'Book - Draft 3' == 'Book - Draft 5'."""
    return re.sub(r"\d+", "#", line.strip())


def find_boilerplate_lines(pages: List[Document]) -> set:
    """Line patterns that repeat on many pages (running headers/footers)."""

    if len(pages) < 5:
        return set()

    counts = Counter()
    for page in pages:
        keys = {_line_key(line) for line in normalize(page.page_content).split("\n")}
        counts.update(key for key in keys if len(key) > 3)

    threshold = max(3, int(len(pages) * BOILERPLATE_RATIO))
    return {key for key, n in counts.items() if n >= threshold}


def clean_page_text(text: str, boilerplate: AbstractSet[str] = frozenset()) -> str:
    lines = [
        line.strip()
        for line in normalize(text).split("\n")
        if line.strip() and _line_key(line) not in boilerplate
    ]
    cleaned = " ".join(lines)
    # The same header can also be glued onto a content line; strip it there too.
    for key in boilerplate:
        pattern = r"\s*".join(re.escape(part) for part in key.split()).replace("\\#", r"\d*")
        cleaned = re.sub(pattern, " ", cleaned)
    return " ".join(cleaned.split())


def file_slug(file_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", file_name)


def make_chunk_id(file_name: str, page: int, index: int, text: str) -> str:
    """Stable ID: '<file-slug>#p<page>#c<index>#<content-hash>'.

    The file prefix lets the indexer list and delete one document's vectors;
    the content hash changes when the text changes, so stale vectors are
    detected on re-index.
    """

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{file_slug(file_name)}#p{page}#c{index}#{digest}"


def chunk_documents(pages: List[Document]) -> List[Document]:
    boilerplate = find_boilerplate_lines(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[". ", "; ", ", ", " ", ""],
    )

    chunks: List[Document] = []

    for page in pages:
        text = clean_page_text(page.page_content, boilerplate)

        if len(text) < MIN_PAGE_CHARS:
            continue

        pieces = [p.strip() for p in splitter.split_text(text)]
        pieces = [p for p in pieces if len(p) >= MIN_CHUNK_CHARS]

        for index, piece in enumerate(pieces):
            metadata = dict(page.metadata)
            metadata["chunk_index"] = index
            metadata["chunk_id"] = make_chunk_id(
                metadata["file_name"], metadata["page"], index, piece
            )
            chunks.append(Document(page_content=piece, metadata=metadata))

    return chunks


if __name__ == "__main__":
    from src.ingestion.loader import list_documents, load_document

    for file in list_documents():
        pages = load_document(file)
        chunks = chunk_documents(pages)
        print(f"{file.name}: {len(pages)} pages -> {len(chunks)} chunks")
        print("Boilerplate removed:", sorted(find_boilerplate_lines(pages)))
        if chunks:
            print("First chunk:", chunks[0].metadata["chunk_id"])
            print(chunks[0].page_content[:300])
