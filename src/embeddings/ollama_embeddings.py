import time

from langchain_ollama import OllamaEmbeddings

from config import EMBEDDING_MODEL


class QwenEmbeddings:
    """
    Qwen embedding client using Ollama.
    """

    def __init__(
        self,
        model_name=EMBEDDING_MODEL,
        base_url="http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url

        self.embeddings = OllamaEmbeddings(
            model=self.model_name,
            base_url=self.base_url,
        )

    def get_base_url(self):
        """Return the Ollama server URL."""
        return self.base_url

    def embed_query(self, text):
        """Generate an embedding for a single query."""

        if not text or not text.strip():
            raise ValueError(
                "Cannot create embedding from empty text."
            )

        return self.embeddings.embed_query(text)

    def embed_documents(self, texts):
        """Generate embeddings for multiple documents."""

        if not texts:
            return []

        return self.embeddings.embed_documents(texts)


if __name__ == "__main__":

    print("=" * 60)
    print("QWEN EMBEDDING TEST")
    print("=" * 60)

    try:

        embeddings = QwenEmbeddings()

        text = (
            "Machine learning is a field of "
            "artificial intelligence."
        )

        start_time = time.perf_counter()

        print("Generating embedding...")

        vector = embeddings.embed_query(text)

        elapsed = time.perf_counter() - start_time

        print("Embedding generated successfully")

        print(f"Model: {embeddings.model_name}")

        print(f"Dimension: {len(vector)}")

        print("First 10 values:")

        print(vector[:10])

        print(
            f"Embedding time: {elapsed:.4f} seconds"
        )

        print("=" * 60)
        print("EMBEDDING TEST SUCCESSFUL")
        print("=" * 60)

    except Exception as e:

        print("=" * 60)
        print("EMBEDDING TEST FAILED")
        print("=" * 60)

        print(f"Error: {e}")

        raise