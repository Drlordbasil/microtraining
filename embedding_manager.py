import numpy as np
import ollama
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingManager:
    def __init__(self, embedding_model: str = 'mxbai-embed-large'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing EmbeddingManager with model: {embedding_model}")
        
        self.embedding_model = embedding_model
        self.embedding_dim = 1024  # Default dimension, will be updated on first embedding
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        try:
            # Verify the embedding model is available
            self._verify_embedding_model()
            self.logger.info("EmbeddingManager initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing EmbeddingManager: {str(e)}")
            raise

    def _verify_embedding_model(self):
        try:
            # Attempt to get an embedding for a simple text to verify the model
            test_embedding = ollama.embeddings(model=self.embedding_model, prompt="test")
            self.embedding_dim = len(test_embedding['embedding'])
            self.logger.info(f"Embedding model verified. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"Failed to verify embedding model: {str(e)}")
            raise ValueError(f"Invalid or unavailable embedding model: {self.embedding_model}")

    def get_embedding(self, text: str) -> np.ndarray:
        if not text:
            self.logger.warning("Attempted to get embedding for empty text")
            return np.zeros(self.embedding_dim)

        if text not in self.embeddings_cache:
            try:
                response = ollama.embeddings(model=self.embedding_model, prompt=text)
                embedding = np.array(response['embedding'])
                
                if embedding.shape[0] != self.embedding_dim:
                    raise ValueError(f"Unexpected embedding dimension: {embedding.shape[0]}")
                
                self.embeddings_cache[text] = embedding
                self.logger.info(f"Obtained embedding for text: '{text[:50]}...'")
            except Exception as e:
                self.logger.error(f"Error obtaining embedding: {str(e)}")
                raise

        return self.embeddings_cache[text]

    def clear_cache(self):
        self.embeddings_cache.clear()
        self.logger.info("Embedding cache cleared")

    def get_cached_embeddings(self) -> Dict[str, np.ndarray]:
        return self.embeddings_cache

    def set_cached_embeddings(self, cache: Dict[str, np.ndarray]):
        if not isinstance(cache, dict):
            raise ValueError("Cache must be a dictionary")
        
        for text, embedding in cache.items():
            if not isinstance(embedding, np.ndarray) or embedding.shape != (self.embedding_dim,):
                raise ValueError(f"Invalid embedding for text: {text[:50]}...")

        self.embeddings_cache = cache
        self.logger.info(f"Set {len(cache)} cached embeddings")

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def find_closest_embedding(self, target_embedding: np.ndarray, text_list: List[str]) -> str:
        if not isinstance(target_embedding, np.ndarray) or target_embedding.shape != (self.embedding_dim,):
            raise ValueError("Invalid target embedding")
        
        if not text_list:
            raise ValueError("Empty text list provided")

        try:
            return max(text_list, key=lambda x: np.dot(self.get_embedding(x), target_embedding))
        except Exception as e:
            self.logger.error(f"Error finding closest embedding: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Test the EmbeddingManager
        em = EmbeddingManager()
        test_text = "This is a test sentence."
        embedding = em.get_embedding(test_text)
        print(f"Embedding shape: {embedding.shape}")

        # Test finding closest embedding
        text_list = ["Hello world", "Goodbye world", "This is a test"]
        closest = em.find_closest_embedding(embedding, text_list)
        print(f"Closest text to '{test_text}': '{closest}'")

        print("EmbeddingManager tests passed successfully")
    except Exception as e:
        logging.error(f"Error in EmbeddingManager test: {str(e)}")
        raise