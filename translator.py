import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Translator:
    def __init__(self, target_dim):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.target_dim = target_dim
        self.logger.info(f"Initializing Translator with target dimension: {target_dim}")

    def translate(self, embedding):
        try:
            if len(embedding) == self.target_dim:
                return embedding
            
            if len(embedding) > self.target_dim:
                self.logger.debug(f"Truncating embedding from {len(embedding)} to {self.target_dim}")
                return embedding[:self.target_dim]
            
            self.logger.debug(f"Padding embedding from {len(embedding)} to {self.target_dim}")
            return np.pad(embedding, (0, self.target_dim - len(embedding)), 'constant')
        except Exception as e:
            self.logger.error(f"Error in translate method: {str(e)}")
            raise

    def translate_tensor(self, embedding):
        try:
            translated_embedding = self.translate(embedding)
            return torch.FloatTensor(translated_embedding).unsqueeze(0)
        except Exception as e:
            self.logger.error(f"Error in translate_tensor method: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        translator = Translator(target_dim=2048)
        test_embedding = np.random.rand(1024)
        translated_embedding = translator.translate(test_embedding)
        print(f"Original dimension: {len(test_embedding)}, Translated dimension: {len(translated_embedding)}")
        
        tensor_embedding = translator.translate_tensor(test_embedding)
        print(f"Tensor shape: {tensor_embedding.shape}")
    except Exception as e:
        logging.error(f"Error in Translator test: {str(e)}")
        raise