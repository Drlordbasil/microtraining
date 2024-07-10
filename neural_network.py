import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing NeuralNetwork with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
        
        try:
            self._validate_dimensions(input_size, hidden_size, output_size)
            
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.layer2 = nn.Linear(hidden_size, hidden_size)
            self.layer3 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
            
            self.logger.info("NeuralNetwork initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing NeuralNetwork: {str(e)}")
            raise

    def _validate_dimensions(self, input_size, hidden_size, output_size):
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("All dimensions must be positive integers")
        self.logger.debug("Dimensions validated successfully")

    def forward(self, x):
        try:
            self.logger.debug(f"Forward pass input shape: {x.shape}")
            if x.shape[1] != self.input_size:
                raise ValueError(f"Input tensor shape {x.shape} does not match expected input size {self.input_size}")
            
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            x = self.layer3(x)
            
            self.logger.debug(f"Forward pass output shape: {x.shape}")
            return x
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            raise

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def log_model_summary(self):
        self.logger.info("Neural Network Summary:")
        self.logger.info(f"Input size: {self.input_size}")
        self.logger.info(f"Hidden size: {self.hidden_size}")
        self.logger.info(f"Output size: {self.output_size}")
        self.logger.info(f"Total trainable parameters: {self.get_parameter_count()}")

if __name__ == "__main__":
    try:
        input_size, hidden_size, output_size = 2048, 512, 1024
        model = NeuralNetwork(input_size, hidden_size, output_size)
        model.log_model_summary()
        
        test_input = torch.randn(1, input_size)
        output = model(test_input)
        print(f"Test output shape: {output.shape}")
    except Exception as e:
        logging.error(f"Error in NeuralNetwork test: {str(e)}")
        raise