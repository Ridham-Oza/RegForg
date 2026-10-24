from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".md",
    ".csv",
    ".json",
    ".html",
    ".xml",
    ".pptx",
}

DOCUMENT_PATH = Path("Data/documents")


def load_document(file_path):
    """
    Load a document based on its file extension.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {extension}"
        )

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif extension in {".txt", ".md", ".csv", ".json", ".html", ".xml"}:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))

    else:
        raise ValueError(
            f"Loader not implemented for: {extension}"
        )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = str(file_path)
        document.metadata["file_name"] = file_path.name

    return documents


if __name__ == "__main__":

    print("=" * 60)
    print("PDF LOADER TEST")
    print("=" * 60)

    pdf_path = (
        DOCUMENT_PATH / "MachineLearning-3.pdf"
    )

    print(f"File: {pdf_path}")

    try:

        documents = load_document(pdf_path)

        print(
            f"Pages loaded: {len(documents)}"
        )

        if documents:

            print(
                f"First page characters: "
                f"{len(documents[0].page_content)}"
            )

            print(
                f"Source: "
                f"{documents[0].metadata.get('source')}"
            )

            print(
                f"File name: "
                f"{documents[0].metadata.get('file_name')}"
            )

        print("=" * 60)
        print("✅ PDF LOADING SUCCESSFUL")
        print("=" * 60)

    except Exception as e:

        print("=" * 60)
        print("❌ PDF LOADING FAILED")
        print("=" * 60)

        print(f"Error: {e}")

        raise


