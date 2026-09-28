from langchain_ollama import OllamaEmbeddings

from config import (EMBEDDING_MODEL, BASE_URL)

class QwenEmbeddings:

    """ Wrapper around Ollama Qwen Embeddings model. """

    def __init__(self, model: str = EMBEDDING_MODEL, base_url: str = BASE_URL):
        self.model = model
        self.base_url = base_url
        self.embeddings = OllamaEmbeddings(model=self.model, base_url=self.base_url)


    def embed_documents(self, texts):
        """ Generate embeddings for a list of documents. """
        return self.embeddings.embed_documents(texts)

    
    def embed_query(self, text):
        """ Genrate embedding for a single query."""
        return self.embeddings.embed_query(text)

    def get_embedding_dimension(self):
        """ get the dimension dynamically"""
        vector = self.embed_query("dimension_test")

        return len(vector)