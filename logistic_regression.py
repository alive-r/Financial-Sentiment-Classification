import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization_strength=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization_strength = regularization_strength
        self.weights = None
        self.bias = None
        self.unique_classes = None

    def cal_softmax(self, linear_scores):
        shifted_scores = linear_scores - np.max(linear_scores, axis=1, keepdims=True)
        exponentials = np.exp(shifted_scores)
        sum_exponentials = np.sum(exponentials, axis=1, keepdims=True)
        return exponentials / sum_exponentials

    def fit(self, training_features, training_labels):
        self.unique_classes = np.unique(training_labels)
        sample_count = training_features.shape[0]
        feature_count = training_features.shape[1]
        class_count = len(self.unique_classes)
        self.weights = np.random.randn(feature_count, class_count) * 0.01
        self.bias = np.zeros(class_count)
        one_hot_labels = np.zeros((sample_count, class_count))

        for i in range(sample_count):
            current_label = training_labels[i]
            label_index = 0
            for j in range(len(self.unique_classes)):
                if self.unique_classes[j] == current_label:
                    label_index = j
                    break
            one_hot_labels[i, label_index] = 1

        for iteration in range(self.max_iterations):
            linear_output = np.dot(training_features, self.weights) + self.bias
            probs = self.cal_softmax(linear_output)
            prediction_error = probs - one_hot_labels
            weight_gradient = np.dot(training_features.T, prediction_error) / sample_count
            weight_gradient = weight_gradient + (self.regularization_strength * self.weights)
            bias_gradient = np.mean(prediction_error, axis=0)
            self.weights = self.weights - self.learning_rate * weight_gradient
            self.bias = self.bias - self.learning_rate * bias_gradient

        return self

    def predict(self, input_features):
        linear_output = np.dot(input_features, self.weights) + self.bias
        probs = self.cal_softmax(linear_output)
        final_predictions = []
        for probability_vector in probs:
            best_class_index = np.argmax(probability_vector)
            final_predictions.append(self.unique_classes[best_class_index])
        return np.array(final_predictions) 