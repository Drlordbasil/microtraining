import logging
import random
import json
import requests
import os
import sys
import time
from typing import List, Dict, Any, Optional, Union, Callable
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_manager.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class DataManager:
    def __init__(self, dataset_url: str = "https://huggingface.co/datasets/BAAI/Infinity-Instruct/resolve/main/3M/train.jsonl", split: str = 'train', cache_dir: str = '.cache'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing DataManager with dataset URL: {dataset_url}")
        self.dataset_url = dataset_url
        self.split = split
        self.cache_dir = cache_dir
        self.dataset: List[str] = []
        self.train_data: List[str] = []
        self.test_data: List[str] = []
        self.session = self._create_robust_session()

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self.logger.info(f"Created cache directory: {self.cache_dir}")

    def _create_robust_session(self) -> requests.Session:
        """Create a robust session with retry strategy."""
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def load_dataset(self, force_reload: bool = False) -> None:
        """Load the dataset from URL or cache."""
        cache_file = os.path.join(self.cache_dir, 'dataset_cache.json')
        
        if not force_reload and os.path.exists(cache_file):
            self.logger.info("Loading dataset from cache...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
                self.logger.info(f"Dataset loaded from cache. Total entries: {len(self.dataset)}")
                return
            except Exception as e:
                self.logger.error(f"Failed to load dataset from cache: {str(e)}")
                self.logger.info("Falling back to loading from URL...")

        self.logger.info(f"Loading dataset from URL: {self.dataset_url}")
        try:
            response = self.session.get(self.dataset_url, stream=True)
            response.raise_for_status()
            self.dataset = []
            for line in response.iter_lines():
                if line:
                    try:
                        json_obj = json.loads(line)
                        extracted_text = self.extract_text(json_obj)
                        if extracted_text:
                            self.dataset.append(extracted_text)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse JSON line: {line}")
                    except Exception as e:
                        self.logger.error(f"Error processing line: {str(e)}")

            self.logger.info(f"Dataset loaded. Total entries: {len(self.dataset)}")
            
            # Cache the loaded dataset
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f)
            self.logger.info(f"Dataset cached to {cache_file}")

        except requests.RequestException as e:
            self.logger.error(f"Network error while fetching dataset: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while loading dataset: {str(e)}")
            raise

    def extract_text(self, entry: Dict[str, Any]) -> Optional[str]:
        """Extract text from a dataset entry."""
        try:
            if not isinstance(entry, dict):
                self.logger.warning(f"Entry is not a dictionary: {entry}")
                return None

            if 'conversations' in entry and isinstance(entry['conversations'], list):
                conversations = entry['conversations']
                if conversations and isinstance(conversations[0], dict) and 'value' in conversations[0]:
                    return conversations[0]['value']
            elif 'text' in entry:
                return entry['text']
            elif 'sentence' in entry:
                return entry['sentence']
            elif 'target_value' in entry:
                return entry['target_value']
            else:
                self.logger.warning(f"Unexpected entry structure: {entry}")
                return str(entry)
        except Exception as e:
            self.logger.error(f"Error extracting text from entry: {str(e)}")
            return None

    def sample_dataset(self, sample_fraction: float = 0.1) -> List[str]:
        """Sample the dataset."""
        if not self.dataset:
            self.logger.warning("Dataset not loaded. Attempting to load...")
            self.load_dataset()
        
        try:
            sample_size = max(1, int(len(self.dataset) * sample_fraction))
            return random.sample(self.dataset, sample_size)
        except Exception as e:
            self.logger.error(f"Error sampling dataset: {str(e)}")
            raise

    def prepare_data(self, sample_fraction: float = 0.001, test_size: float = 0.2) -> None:
        """Prepare train and test data."""
        self.logger.info(f"Preparing data with sample_fraction={sample_fraction}, test_size={test_size}")
        try:
            all_sentences = self.sample_dataset(sample_fraction)
            if not all_sentences:
                raise ValueError("No data available after sampling.")
            self.train_data, self.test_data = train_test_split(all_sentences, test_size=test_size, random_state=42)
            self.logger.info(f"Data prepared. Training set size: {len(self.train_data)}, Test set size: {len(self.test_data)}")
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def get_train_data(self) -> List[str]:
        """Get training data."""
        if not self.train_data:
            self.logger.warning("Train data is empty. Make sure to prepare data first.")
        return self.train_data

    def get_test_data(self) -> List[str]:
        """Get test data."""
        if not self.test_data:
            self.logger.warning("Test data is empty. Make sure to prepare data first.")
        return self.test_data

    def set_train_data(self, train_data: List[str]) -> None:
        """Set training data."""
        if not isinstance(train_data, list):
            raise ValueError("train_data must be a list")
        self.train_data = train_data
        self.logger.info(f"Loaded {len(self.train_data)} training samples.")

    def add_to_train_data(self, sentences: Union[str, List[str]]) -> None:
        """Add sentences to training data."""
        if isinstance(sentences, str):
            sentences = [sentences]
        self.train_data.extend(sentences)
        self.logger.info(f"Added {len(sentences)} new sentence(s) to training data. New training data size: {len(self.train_data)}")

    def add_to_test_data(self, sentences: Union[str, List[str]]) -> None:
        """Add sentences to test data."""
        if isinstance(sentences, str):
            sentences = [sentences]
        self.test_data.extend(sentences)
        self.logger.info(f"Added {len(sentences)} new sentence(s) to test data. New test data size: {len(self.test_data)}")

    def clear_data(self) -> None:
        """Clear all data."""
        self.dataset = []
        self.train_data = []
        self.test_data = []
        self.logger.info("Cleared all data.")

    def get_data_stats(self) -> Dict[str, int]:
        """Get statistics about the data."""
        stats = {
            "total_dataset_size": len(self.dataset),
            "train_data_size": len(self.train_data),
            "test_data_size": len(self.test_data),
        }
        self.logger.info(f"Data stats: {stats}")
        return stats

    def process_data_in_parallel(self, func: Callable[[str], Any], data: List[str], max_workers: int = 4) -> List[Any]:
        """Process data in parallel."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(func, item): item for item in data}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing item {item}: {str(e)}")
        return results

    @staticmethod
    def validate_data_integrity(data: List[str]) -> bool:
        """Validate the integrity of the data."""
        return all(isinstance(item, str) and item.strip() for item in data)

    def perform_data_augmentation(self, sentences: List[str], augmentation_factor: int = 2) -> List[str]:
        """Perform simple data augmentation by shuffling words."""
        augmented_data = []
        for sentence in sentences:
            words = sentence.split()
            for _ in range(augmentation_factor):
                random.shuffle(words)
                augmented_data.append(" ".join(words))
        return augmented_data

    def save_data_to_file(self, data: List[str], filename: str) -> None:
        """Save data to a file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(f"{item}\n")
            self.logger.info(f"Data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving data to file: {str(e)}")
            raise

    def load_data_from_file(self, filename: str) -> List[str]:
        """Load data from a file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = [line.strip() for line in f]
            self.logger.info(f"Data loaded from {filename}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data from file: {str(e)}")
            raise

if __name__ == "__main__":
    # Test the DataManager
    try:
        print("Initializing DataManager...")
        dm = DataManager()
        
        print("Loading dataset...")
        dm.load_dataset()
        
        print("Preparing data...")
        dm.prepare_data(sample_fraction=0.001)
        
        print("Data statistics:")
        print(dm.get_data_stats())
        
        print("Adding new data...")
        dm.add_to_train_data("This is a new training sentence.")
        dm.add_to_test_data("This is a new test sentence.")
        
        print("Updated data statistics:")
        print(dm.get_data_stats())
        
        print("Validating data integrity...")
        train_data_valid = dm.validate_data_integrity(dm.get_train_data())
        test_data_valid = dm.validate_data_integrity(dm.get_test_data())
        print(f"Train data integrity: {'Valid' if train_data_valid else 'Invalid'}")
        print(f"Test data integrity: {'Valid' if test_data_valid else 'Invalid'}")
        
        print("Performing data augmentation...")
        augmented_data = dm.perform_data_augmentation(dm.get_train_data()[:5], augmentation_factor=2)
        print(f"Augmented data sample: {augmented_data[:2]}")
        
        print("Saving and loading data...")
        dm.save_data_to_file(dm.get_train_data()[:100], "sample_train_data.txt")
        loaded_data = dm.load_data_from_file("sample_train_data.txt")
        print(f"Loaded data sample: {loaded_data[:2]}")
        
        print("Processing data in parallel...")
        def uppercase_text(text):
            return text.upper()
        processed_data = dm.process_data_in_parallel(uppercase_text, dm.get_train_data()[:10])
        print(f"Processed data sample: {processed_data[:2]}")
        
        print("DataManager tests completed successfully.")
    except Exception as e:
        print(f"Error in DataManager test: {str(e)}")
        logging.error(f"Error in DataManager test: {str(e)}", exc_info=True)
        raise