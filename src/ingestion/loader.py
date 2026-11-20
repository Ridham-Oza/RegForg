"""
Document loading.

Each loaded page becomes a LangChain `Document` with metadata:
    source     absolute path of the file
    file_name  base name of the file
    page       1-based page number (matches what a PDF viewer shows)
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

from config import DOCUMENTS_DIR

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(file_path) -> List[Document]:
    """Load one file into page-level documents."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    base = {"source": str(path.resolve()), "file_name": path.name}

    if extension == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={**base, "page": number},
            )
            for number, page in enumerate(reader.pages, start=1)
        ]

    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={**base, "page": 1})]


def list_documents(directory=DOCUMENTS_DIR) -> List[Path]:
    """Return every supported file in the documents directory."""

    directory = Path(directory)

    if not directory.exists():
        return []

    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


if __name__ == "__main__":
    for file in list_documents():
        pages = load_document(file)
        print(f"{file.name}: {len(pages)} pages")
