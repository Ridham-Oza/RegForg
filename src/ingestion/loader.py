from pathlib import Path

from langchain_community.document_loaders import ( PyPDFLoader, TextLoader, Docx2txtLoader
                                                  )

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".csv", ".json", ".html", ".xml", ".pptx"}

Document_path = "Data/documents"

def load_document(file_path):
    """ Load a document based on the file extension."""

    file_path = Path(Document_path)
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {extension}")

    if extension == ".pdf":
        loader = PyPDFLoader((file_path))
    elif extension == ".txt":
        loader = TextLoader((file_path))
    elif extension == ".docx":
        loader = Docx2txtLoader((file_path))
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    return loader.load()


    for document in documents:
        documents.metadata["source"] = path.name
        documents.metadata["Document_path"] = extension
    return documents



