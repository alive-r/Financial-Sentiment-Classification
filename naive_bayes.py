import numpy as np

class NaiveBayes:
    def __init__(self, laplace_smoothing=1.0):
        self.laplace_smoothing = laplace_smoothing
        self.class_priors = None
        self.feature_probs = None 
        self.classes = None

    def fit(self, training_features, training_labels):
        self.classes = np.unique(training_labels)
        class_count = len(self.classes) 
        feature_count = training_features.shape[1]

        self.class_priors = np.zeros(class_count)
        self.feature_probs = np.zeros((class_count, feature_count))
        
        for i in range(class_count):
            label = self.classes[i]
            training_features_current = training_features[training_labels == label]
            count = len(training_features_current)
            self.class_priors[i] = np.log(count / len(training_labels))
            word_counts = training_features_current.sum(axis=0) + self.laplace_smoothing
            total_counts = word_counts.sum()
            self.feature_probs[i] = np.log(word_counts / total_counts)
        return self

    def predict(self, training_features):
        predictions = []
        scores = training_features @ self.feature_probs.T + self.class_priors
        for score in scores:
            best_index = np.argmax(score)
            predictions.append(self.classes[best_index])
            
        return np.array(predictions)
