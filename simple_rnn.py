import numpy as np

class RecurrentNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, max_epochs=100):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights_input_hidden = np.random.randn(hidden_dim, input_dim) * 0.01
        self.weights_hidden_hidden = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.weights_hidden_output = np.random.randn(output_dim, hidden_dim) * 0.01
        self.bias_hidden = np.zeros((hidden_dim, 1))
        self.bias_output = np.zeros((output_dim, 1))
    
    def compute_tanh(self, x):
        return np.tanh(x) 
    
    def compute_softmax(self, x):
        shifted_x = x - np.max(x)
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x
    
    def forward_pass(self, input_sequence):
        current_hidden = np.zeros((self.hidden_dim, 1))
        for i in range(len(input_sequence)):
            current_input = input_sequence[i]
            term1 = np.dot(self.weights_input_hidden, current_input)
            term2 = np.dot(self.weights_hidden_hidden, current_hidden)
            input = term1 + term2 + self.bias_hidden
            current_hidden = self.compute_tanh(input)
        output = np.dot(self.weights_hidden_output, current_hidden) + self.bias_output
        return self.compute_softmax(output), current_hidden
    
    def predict(self, sequence_batch):
        pred_list = []
        for sequence in sequence_batch:
            inputs = []
            for item in sequence:
                reshaped_item = item.reshape(-1, 1)
                inputs.append(reshaped_item)
            probs, _ = self.forward_pass(inputs)
            best_class = np.argmax(probs)
            pred_list.append(best_class)
        return np.array(pred_list)