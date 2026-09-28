import os
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s:'
)


list_of_files = [

    "app.py",
    "config.py",
    "requirements.txt",
    "README.md",
    ".gitignore",

    "data/documents/.gitkeep",
    

    "src/__init__.py",

    "src/ingestion/__init__.py",
    "src/ingestion/loader.py",
    "src/ingestion/chunker.py",
    "src/ingestion/indexer.py",

    "src/embeddings/__init__.py",
    "src/embeddings/ollama_embeddings.py",

    "src/retrieval/__init__.py",
    "src/retrieval/retriever.py",

    "src/generation/__init__.py",
    "src/generation/llm.py",

    "src/pipeline/__init__.py",
    "src/pipeline/rag_pipeline.py",

    "src/utils/__init__.py",
    "src/utils/metrics.py",
    "src/utils/logger.py",

    "ui/__init__.py",
    "ui/components.py",
    "ui/gradio_app.py",
]


for filepath in list_of_files:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

        logging.info(
            f"Creating directory: {filedir} "
            f"for the file: {filename}"
        )

    if (
        not os.path.exists(filepath)
        or os.path.getsize(filepath) == 0
    ):

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as f:
            pass

        logging.info(
            f"Creating empty file: {filepath}"
        )

    else:

        logging.info(
            f"{filename} already exists"
        )


logging.info(
    "=============================================="
)

logging.info(
    "Qwen 8B RAG project structure created!"
)

logging.info(
    "=============================================="
)