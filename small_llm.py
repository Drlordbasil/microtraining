import torch
import os
import traceback
import logging
from neural_network import NeuralNetwork
from embedding_manager import EmbeddingManager
from training_manager import TrainingManager
from evaluation_manager import EvaluationManager
from data_manager import DataManager
from translator import Translator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmallLLM:
    def __init__(self, model_path='llm_model.pth', embedding_model='mxbai-embed-large', hidden_size=512, learning_rate=0.001):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing SmallLLM...")
        
        try:
            self.model_path = model_path
            self.embedding_manager = EmbeddingManager(embedding_model)
            self.embedding_dim = self.embedding_manager.get_embedding_dim()
            self.logger.info(f"Actual embedding dimension: {self.embedding_dim}")
            
            if self.embedding_dim <= 0:
                raise ValueError(f"Invalid embedding dimension: {self.embedding_dim}")
            
            self.input_dim = self.embedding_dim * 2
            self.translator = Translator(target_dim=self.input_dim)
            
            self.model = NeuralNetwork(self.input_dim, hidden_size, self.embedding_dim)
            self.training_manager = TrainingManager(self.model, self.embedding_manager, learning_rate, self.translator)
            self.evaluation_manager = EvaluationManager(self.model, self.embedding_manager, self.translator)
            self.data_manager = DataManager()
            
            self.logger.info("SmallLLM initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error during SmallLLM initialization: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def train(self, sample_fraction=0.1, iterations=1000):
        self.logger.info(f"Starting training with sample_fraction={sample_fraction}, iterations={iterations}")
        try:
            self.data_manager.prepare_data(sample_fraction)
            train_data = self.data_manager.get_train_data()
            
            if not train_data:
                raise ValueError("No training data available after preparation.")
            
            self.logger.info(f"Train data size: {len(train_data)}")
            self.training_manager.train_continuously(train_data, iterations, self.save_model)
            self.logger.info("Training completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def evaluate(self):
        self.logger.info("Starting model evaluation...")
        try:
            test_data = self.data_manager.get_test_data()
            train_data = self.data_manager.get_train_data()
            
            if not test_data or not train_data:
                raise ValueError("Test data or train data is empty. Cannot evaluate model.")
            
            result = self.evaluation_manager.evaluate_model(test_data, train_data)
            self.logger.info(f"Evaluation result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def predict(self, input_sentence):
        self.logger.info(f"Predicting for input: '{input_sentence[:50]}...'")
        try:
            if not input_sentence:
                raise ValueError("Input sentence is empty.")
            
            train_data = self.data_manager.get_train_data()
            if not train_data:
                self.logger.warning("No training data available.")
                return "I don't have any training data to make predictions. Please train me first."
            
            result = self.evaluation_manager.predict(input_sentence, train_data)
            self.logger.info(f"Prediction result: '{result[:50]}...'")
            return result
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "An error occurred during prediction."

    def save_model(self):
        self.logger.info(f"Saving model to {self.model_path}")
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'training_state': self.training_manager.get_training_state(),
                'embeddings_cache': self.embedding_manager.get_cached_embeddings(),
                'train_data': self.data_manager.get_train_data(),
                'embedding_dim': self.embedding_dim,
                'input_dim': self.input_dim,
            }, self.model_path)
            self.logger.info(f"Model saved successfully at {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def load_model(self):
        self.logger.info(f"Loading model from {self.model_path}")
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"No model found at {self.model_path}")
            
            checkpoint = torch.load(self.model_path)
            
            self.embedding_dim = checkpoint.get('embedding_dim', self.embedding_dim)
            self.input_dim = checkpoint.get('input_dim', self.input_dim)
            self.logger.info(f"Loaded embedding dimension: {self.embedding_dim}, Input dimension: {self.input_dim}")
            
            if self.embedding_dim <= 0 or self.input_dim <= 0:
                raise ValueError(f"Invalid dimensions: embedding_dim={self.embedding_dim}, input_dim={self.input_dim}")
            
            self.model = NeuralNetwork(self.input_dim, self.model.hidden_size, self.embedding_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.training_manager.set_training_state(checkpoint['training_state'])
            self.embedding_manager.set_cached_embeddings(checkpoint['embeddings_cache'])
            
            if 'train_data' in checkpoint:
                self.data_manager.set_train_data(checkpoint['train_data'])
                self.logger.info(f"Loaded {len(checkpoint['train_data'])} training samples.")
            else:
                self.logger.warning("No training data found in the model checkpoint.")
            
            self.logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def add_feedback(self, input_sentence, correct_response, reward):
        self.logger.info(f"Adding feedback for input: '{input_sentence[:50]}...'")
        try:
            if not input_sentence or not correct_response:
                raise ValueError("Input sentence or correct response is empty.")
            
            if reward > 0:
                self.data_manager.add_to_train_data(correct_response)
                loss = self.training_manager.train_on_sentence(correct_response)
                self.logger.info(f"Positive feedback received. Loss: {loss:.4f}")
            else:
                improved_response = self.evaluation_manager.generate_improved_response(input_sentence, correct_response)
                self.data_manager.add_to_train_data(improved_response)
                loss = self.training_manager.train_on_sentence(improved_response)
                self.logger.info(f"Negative feedback received. Generated better response. Loss: {loss:.4f}")
        except Exception as e:
            self.logger.error(f"Error adding feedback: {str(e)}")
            self.logger.error(traceback.format_exc())

    def demonstrate_continuous_learning(self):
        self.logger.info("\nDemonstrating continuous learning:")
        try:
            input_sentence = "Explain the concept of machine learning in simple terms."
            self.logger.info(f"Initial input: '{input_sentence}'")
            response = self.predict(input_sentence)
            self.logger.info(f"Initial response: '{response}'")

            self.add_feedback(input_sentence, response, reward=-1)
            improved_response = self.predict(input_sentence)
            self.logger.info(f"Improved response: '{improved_response}'")
        except Exception as e:
            self.logger.error(f"Error during continuous learning demonstration: {str(e)}")
            self.logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    try:
        logger.info("Starting SmallLLM main execution...")
        llm = SmallLLM()
        llm.train(sample_fraction=0.01, iterations=1000)
        llm.evaluate()
        llm.demonstrate_continuous_learning()
        llm.save_model()
        logger.info("Final model saved. Execution completed successfully.")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())