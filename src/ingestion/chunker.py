from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# Repeated PDF noise
# ---------------------------------------------------------

NOISE_PATTERNS = [
    "The Hundred-Page Machine Learning Book Andriy Burkov",
    "Andriy Burkov's",
]


def clean_page_text(text: str) -> str:
    """
    Remove repeated PDF headers/footers and normalize whitespace.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        # Remove exact repeated book header/footer
        if line.lower() in {
            "the hundred-page machine learning book",
            "andriy burkov",
            "andriy burkov's",
        }:
            continue

        lines.append(line)

    cleaned = "\n".join(lines)

    # Remove repeated spaces
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()


def chunk_documents(
    documents: List[Document],
) -> List[Document]:

    cleaned_documents = []

    # -----------------------------------------------------
    # 1. Clean every PDF page
    # -----------------------------------------------------

    for document in documents:

        cleaned_text = clean_page_text(
            document.page_content
        )

        # Ignore completely empty pages
        if not cleaned_text:
            continue

        # Ignore extremely small text fragments
        if len(cleaned_text) < 80:
            continue

        cleaned_document = Document(
            page_content=cleaned_text,
            metadata=document.metadata.copy(),
        )

        cleaned_documents.append(
            cleaned_document
        )

    # -----------------------------------------------------
    # 2. Split into meaningful chunks
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        cleaned_documents
    )

    # -----------------------------------------------------
    # 3. Final quality filter
    # -----------------------------------------------------

    final_chunks = []

    for chunk in chunks:

        text = chunk.page_content.strip()

        # Ignore very small chunks
        if len(text) < 150:
            continue

        # Ignore chunks that contain only book title noise
        normalized = (
            text
            .replace(" ", "")
            .replace("-", "")
            .lower()
        )

        if normalized.startswith(
            "thehundredpagemachinelearningbook"
        ):
            continue

        chunk.page_content = text

        final_chunks.append(chunk)

    return final_chunks


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from src.ingestion.loader import load_document

    print("=" * 60)
    print("DOCUMENT CHUNKING TEST")
    print("=" * 60)

    documents = load_document(
        "Data/documents/MachineLearning-3.pdf"
    )

    print(
        f"Pages loaded: {len(documents)}"
    )

    chunks = chunk_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    if chunks:

        print("\nFirst chunk:")
        print("-" * 60)
        print(
            chunks[0].page_content[:1000]
        )

        print("\nFirst chunk metadata:")
        print(
            chunks[0].metadata
        )

    print("=" * 60)
    print("✅ CHUNKING SUCCESSFUL")
    print("=" * 60)
    