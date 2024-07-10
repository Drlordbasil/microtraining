import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
from collections import deque
from scipy.spatial.distance import cosine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingManager:
    def __init__(self, model, embedding_manager, learning_rate, translator, memory_size=1000):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing TrainingManager")
        
        self.model = model
        self.embedding_manager = embedding_manager
        self.translator = translator
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.train_data = []
        self.memory = deque(maxlen=memory_size)  # Long-term memory
        self.working_memory = []  # Short-term working memory
        self.learned_data = set()
        self.confidence = 0
        self.competency = 0
        self.reasoning_ability = 0
        self.learning_stage = "Peak of Mount Stupid"
        self.error_history = []
        self.stage_benchmarks = self.initialize_benchmarks()
        self.benchmark_scores = {stage: 0 for stage in self.stage_benchmarks.keys()}

        self.logger.info("TrainingManager initialized successfully")

    def initialize_benchmarks(self):
        return {
            "Peak of Mount Stupid": [
                ("What is 2 + 2?", "4", 0.9),
                ("Name a primary color.", "Red, Blue, or Yellow", 0.9),
                ("What is the capital of the United States?", "Washington D.C.", 0.9)
            ],
            "Valley of Despair": [
                ("Explain the concept of gravity in simple terms.", "Gravity is a force that attracts objects towards each other, with larger objects having stronger gravitational pull.", 0.7),
                ("What is the difference between weather and climate?", "Weather refers to day-to-day conditions, while climate describes long-term patterns in an area.", 0.7),
                ("How does photosynthesis work?", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.", 0.7)
            ],
            "Slope of Enlightenment": [
                ("Describe the impact of the Industrial Revolution on society.", "The Industrial Revolution led to significant social and economic changes, including urbanization, technological advancements, and shifts in labor practices.", 0.8),
                ("Explain the concept of cognitive dissonance in psychology.", "Cognitive dissonance is the mental discomfort experienced when a person holds contradictory beliefs, ideas, or values, often leading to changes in attitudes or behaviors to reduce this discomfort.", 0.8),
                ("How does the theory of evolution by natural selection work?", "Evolution by natural selection occurs when organisms with beneficial traits are more likely to survive and reproduce, passing these traits to offspring, leading to changes in populations over time.", 0.8)
            ],
            "Plateau of Sustainability": [
                ("Analyze the potential long-term effects of artificial intelligence on the job market.", "AI could lead to job displacement in certain sectors, creation of new job types, increased productivity, and a shift towards more creative and interpersonal skills in the workforce. It may require widespread retraining and education adaptation.", 0.9),
                ("Compare and contrast the philosophical ideas of determinism and free will.", "Determinism posits that all events are caused by prior events, while free will suggests that individuals have control over their choices. This debate touches on issues of moral responsibility, the nature of consciousness, and the implications for legal and ethical systems.", 0.9),
                ("Explain the concept of quantum entanglement and its implications for our understanding of reality.", "Quantum entanglement is a phenomenon where particles become interconnected and the quantum state of each particle cannot be described independently. This challenges our classical understanding of reality, suggesting instantaneous communication or shared information across distances, with implications for physics, computing, and philosophy.", 0.9)
            ]
        }

    def train_on_sentence(self, sentence, context=None):
        try:
            if sentence in self.learned_data:
                self.logger.info(f"Sentence already learned: '{sentence[:50]}...'")
                return 0

            input_embedding = self.embedding_manager.get_embedding(sentence)
            if context:
                context_embedding = self.embedding_manager.get_embedding(context)
                input_embedding = np.concatenate([input_embedding, context_embedding])
            
            input_embedding = self.translator.translate(input_embedding)

            target_embedding = input_embedding[:self.embedding_manager.get_embedding_dim()]  # Use only the first half as target

            input_tensor = torch.FloatTensor(input_embedding).unsqueeze(0)
            target_tensor = torch.FloatTensor(target_embedding).unsqueeze(0)

            self.optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()

            self.learned_data.add(sentence)
            self.update_cognitive_metrics(loss.item())
            self.memory.append((sentence, context, loss.item()))
            self.logger.info(f"Trained on sentence: '{sentence[:50]}...', Loss: {loss.item():.4f}")
            return loss.item()
        except Exception as e:
            self.logger.error(f"Error in train_on_sentence: {str(e)}")
            raise

    def update_cognitive_metrics(self, loss):
        try:
            self.competency += 0.01
            self.reasoning_ability += 0.005 * (1 - loss)
            
            self.perform_benchmark_tests()
            self.update_learning_stage()

            self.confidence = max(0, min(1, self.confidence))
            self.competency = max(0, min(1, self.competency))
            self.reasoning_ability = max(0, min(1, self.reasoning_ability))
            self.logger.info(f"Updated Confidence: {self.confidence:.2f}, Competency: {self.competency:.2f}, Reasoning: {self.reasoning_ability:.2f}, Stage: {self.learning_stage}")
        except Exception as e:
            self.logger.error(f"Error in update_cognitive_metrics: {str(e)}")
            raise

    def perform_benchmark_tests(self):
        try:
            for stage, benchmarks in self.stage_benchmarks.items():
                total_score = 0
                for question, expected_answer, threshold in benchmarks:
                    generated_answer = self.generate_reasoning(question, expected_answer)
                    similarity = self.compute_similarity(generated_answer, expected_answer)
                    total_score += similarity / len(benchmarks)
                self.benchmark_scores[stage] = total_score
                self.logger.info(f"Benchmark score for {stage}: {total_score:.2f}")
        except Exception as e:
            self.logger.error(f"Error in perform_benchmark_tests: {str(e)}")
            raise

    def update_learning_stage(self):
        try:
            current_stage_index = list(self.stage_benchmarks.keys()).index(self.learning_stage)
            if current_stage_index < len(self.stage_benchmarks) - 1:
                next_stage = list(self.stage_benchmarks.keys())[current_stage_index + 1]
                if self.benchmark_scores[next_stage] > 0.7:  # Threshold for advancing to next stage
                    self.learning_stage = next_stage
                    self.logger.info(f"Advanced to new learning stage: {self.learning_stage}")
            
            self.confidence = self.benchmark_scores[self.learning_stage]
        except Exception as e:
            self.logger.error(f"Error in update_learning_stage: {str(e)}")
            raise

    def compute_similarity(self, generated, expected):
        try:
            generated_embedding = self.embedding_manager.get_embedding(generated)
            expected_embedding = self.embedding_manager.get_embedding(expected)
            
            similarity = 1 - cosine(generated_embedding, expected_embedding)
            
            similarity = max(0, min(1, similarity))
            
            return similarity
        except Exception as e:
            self.logger.error(f"Error in compute_similarity: {str(e)}")
            raise

    def train_continuously(self, train_data, iterations=1000, save_callback=None):
        self.logger.info("Starting continuous training...")
        try:
            self.train_data = train_data
            self.logger.info(f"Training data size: {len(self.train_data)}")

            for iteration in tqdm(range(1, iterations + 1), desc="Training"):
                unlearned_data = [s for s in self.train_data if s not in self.learned_data]
                if not unlearned_data:
                    self.logger.info("All data learned. Initiating reasoning and generalization phase.")
                    self.reasoning_phase(iterations - iteration)
                    break

                sentence = random.choice(unlearned_data)
                context = self.generate_context()
                loss = self.train_on_sentence(sentence, context)
                self.error_history.append(loss)

                if iteration % 100 == 0:
                    self.reflect_and_adjust()
                    if save_callback:
                        save_callback()
        except Exception as e:
            self.logger.error(f"Error in train_continuously: {str(e)}")
            raise

    def generate_context(self):
        try:
            if len(self.working_memory) > 0 and random.random() < 0.7:  # 70% chance to use working memory
                return random.choice(self.working_memory)
            elif len(self.memory) > 0 and random.random() < 0.3:  # 30% chance to use long-term memory
                return random.choice(self.memory)[0]
            return None
        except Exception as e:
            self.logger.error(f"Error in generate_context: {str(e)}")
            raise

    def reflect_and_adjust(self):
        try:
            recent_errors = self.error_history[-100:]
            avg_error = sum(recent_errors) / len(recent_errors)
            if avg_error > 0.5:  # High error rate
                self.logger.info("High error rate detected. Adjusting learning strategy...")
                self.optimizer.param_groups[0]['lr'] *= 0.9  # Reduce learning rate
                self.working_memory = list(self.memory)[-10:]  # Focus on recent memories
            elif avg_error < 0.1:  # Low error rate
                self.logger.info("Low error rate detected. Increasing complexity...")
                self.optimizer.param_groups[0]['lr'] *= 1.1  # Increase learning rate
                self.generate_complex_examples()
        except Exception as e:
            self.logger.error(f"Error in reflect_and_adjust: {str(e)}")
            raise

    def generate_complex_examples(self):
        try:
            if len(self.memory) < 2:
                return
            example1, example2 = random.sample(list(self.memory), 2)
            combined = self.generate_reasoning(example1[0], example2[0])
            self.train_data.append(combined)
            self.logger.info(f"Generated complex example: {combined}")
        except Exception as e:
            self.logger.error(f"Error in generate_complex_examples: {str(e)}")
            raise

    def reasoning_phase(self, iterations):
        self.logger.info("Entering reasoning phase...")
        try:
            for _ in range(iterations):
                if len(self.memory) < 2:
                    break
                premise, conclusion = random.sample(list(self.memory), 2)
                reasoning = self.generate_reasoning(premise[0], conclusion[0])
                self.train_on_sentence(reasoning, context=f"{premise[0]} -> {conclusion[0]}")
        except Exception as e:
            self.logger.error(f"Error in reasoning_phase: {str(e)}")
            raise

    def generate_reasoning(self, premise, conclusion):
        try:
            premise_embedding = self.embedding_manager.get_embedding(premise)
            conclusion_embedding = self.embedding_manager.get_embedding(conclusion)
            
            combined_embedding = np.concatenate([premise_embedding, conclusion_embedding])
            input_tensor = torch.FloatTensor(combined_embedding).unsqueeze(0)
            
            with torch.no_grad():
                output_embedding = self.model(input_tensor).squeeze(0).numpy()
            
            if not self.memory:
                self.logger.warning("Memory is empty, cannot find closest sentence.")
                return f"Given that {premise}, we can infer {conclusion}."

            closest_sentence = max(self.memory, key=lambda x: np.dot(self.embedding_manager.get_embedding(x[0]), output_embedding))
            
            reasoning = f"Given that {premise}, we can infer {conclusion}. This is because {closest_sentence[0]}, which connects these concepts."
            
            return reasoning
        except Exception as e:
            self.logger.error(f"Error in generate_reasoning: {str(e)}")
            raise

    def get_training_state(self):
        return {
            'learned_data': list(self.learned_data),
            'memory': list(self.memory),
            'working_memory': self.working_memory,
            'confidence': self.confidence,
            'competency': self.competency,
            'reasoning_ability': self.reasoning_ability,
            'learning_stage': self.learning_stage,
            'error_history': self.error_history,
            'benchmark_scores': self.benchmark_scores
        }

    def set_training_state(self, state):
        try:
            self.learned_data = set(state['learned_data'])
            self.memory = deque(state['memory'], maxlen=self.memory.maxlen)
            self.working_memory = state['working_memory']
            self.confidence = state['confidence']
            self.competency = state['competency']
            self.reasoning_ability = state['reasoning_ability']
            self.learning_stage = state['learning_stage']
            self.error_history = state['error_history']
            self.benchmark_scores = state['benchmark_scores']
            self.logger.info("Training state set successfully")
        except Exception as e:
            self.logger.error(f"Error in set_training_state: {str(e)}")
            raise

    def continuous_learning(self, new_sentence, context=None):
        try:
            if new_sentence not in self.learned_data:
                loss = self.train_on_sentence(new_sentence, context)
                self.train_data.append(new_sentence)
                self.working_memory.append(new_sentence)
                if len(self.working_memory) > 10:  # Limit working memory size
                    self.working_memory.pop(0)
                return loss
            else:
                self.logger.info(f"Sentence already learned: '{new_sentence[:50]}...'")
                return 0
        except Exception as e:
            self.logger.error(f"Error in continuous_learning: {str(e)}")
            raise

if __name__ == "__main__":
    # This section can be used for testing the TrainingManager independently
    pass