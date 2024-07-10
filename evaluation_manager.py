import torch
import random
import ollama
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EvaluationManager:
    def __init__(self, model, embedding_manager, translator):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing EvaluationManager")
        self.model = model
        self.embedding_manager = embedding_manager
        self.translator = translator

    def predict(self, input_sentence, train_data):
        self.logger.info(f"Predicting for input: '{input_sentence[:50]}...'")
        try:
            if not train_data:
                self.logger.warning("Train data is empty. Cannot make a prediction.")
                return "I don't have enough information to make a prediction."

            input_embedding = self.embedding_manager.get_embedding(input_sentence)
            input_tensor = self.translator.translate_tensor(input_embedding)

            with torch.no_grad():
                output_embedding = self.model(input_tensor).squeeze(0).numpy()

            closest_sentence = max(train_data, key=lambda x: np.dot(self.embedding_manager.get_embedding(x), output_embedding))
            self.logger.info(f"Predicted closest sentence: '{closest_sentence[:50]}...'")
            return closest_sentence
        except Exception as e:
            self.logger.error(f"Error in predict: {str(e)}")
            return "An error occurred while making a prediction."

    def generate_and_rate(self, test_data, train_data):
        self.logger.info("\nGenerating and rating responses...")
        try:
            if not test_data or not train_data:
                self.logger.warning("Test data or train data is empty. Cannot generate and rate responses.")
                return None, None, None

            test_sentence = random.choice(test_data)
            generated_response = self.predict(test_sentence, train_data)
            
            self.logger.info(f"Input: {test_sentence}")
            self.logger.info(f"Generated: {generated_response}")
            
            rating = self.rate_response(test_sentence, generated_response)
            self.logger.info(f"Rating: {rating}/5")

            if rating < 3:
                self.logger.info("Rating below 3. Generating improved response using Ollama...")
                improved_response = self.generate_improved_response(test_sentence, generated_response)
                self.logger.info(f"Improved response: {improved_response}")
                return test_sentence, improved_response, 1
            return None, None, None
        except Exception as e:
            self.logger.error(f"Error in generate_and_rate: {str(e)}")
            return None, None, None

    def rate_response(self, input_sentence, response):
        self.logger.info("Rating response...")
        prompt = (
            "Rate the following response to the input on a scale of 1-5, where 5 is excellent and 1 is poor.\n\n"
            f"Input: {input_sentence}\n"
            f"Response: {response}\n\n"
            "Your task is to provide ONLY a single integer rating (1, 2, 3, 4, or 5) without any additional text or explanation. "
            "Do not include any other words or punctuation in your response.\n\n"
            "Rating:"
        )
        try:
            rating_response = ollama.generate(model='llama3', prompt=prompt)
            rating_text = rating_response['response'].strip()
            
            if rating_text in ['1', '2', '3', '4', '5']:
                rating = int(rating_text)
                self.logger.info(f"Received rating: {rating}")
                return rating
            else:
                raise ValueError(f"Invalid rating received: {rating_text}")
        except Exception as e:
            self.logger.error(f"Error in rating response: {str(e)}")
            self.logger.warning("Defaulting to rating 3 due to error.")
            return 3

    def generate_improved_response(self, input_sentence, original_response):
        self.logger.info("Generating improved response...")
        prompt = f"Given the input: '{input_sentence}', generate a better response than: '{original_response}'"
        try:
            improved_response = ollama.generate(model='llama3', prompt=prompt)
            return improved_response['response']
        except Exception as e:
            self.logger.error(f"Error in generating improved response: {str(e)}")
            self.logger.warning("Returning original response due to error.")
            return original_response

    def evaluate_model(self, test_data, train_data, num_samples=100):
        self.logger.info("Evaluating model...")
        try:
            if not test_data or not train_data:
                self.logger.warning("Test data or train data is empty. Cannot evaluate model.")
                return 0

            test_sentences = random.sample(test_data, min(len(test_data), num_samples))
            total_rating = 0
            valid_ratings = 0

            for sentence in test_sentences:
                prediction = self.predict(sentence, train_data)
                rating = self.rate_response(sentence, prediction)
                if rating != 3:  # Only count non-default ratings
                    total_rating += rating
                    valid_ratings += 1

            if valid_ratings == 0:
                self.logger.warning("No valid ratings obtained during evaluation.")
                return 0

            average_rating = total_rating / valid_ratings
            self.logger.info(f"Model average rating: {average_rating:.2f}/5 (based on {valid_ratings} valid ratings)")
            return average_rating
        except Exception as e:
            self.logger.error(f"Error in evaluate_model: {str(e)}")
            return 0