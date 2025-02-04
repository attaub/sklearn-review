"""
base estimator is used for creating
    custom estimators,
    custom models,
    transformers,
    or other components
that integrate seamlessly with the scikit-learn ecosystem.

"""

############################################
# import the necessary classes and functions from sklearn.base

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

############################################
# Create a simple custom classifier implementing a basic decision rule


class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Store the mean of each feature for each class
        self.means_ = np.array(
            [X[y == cls].mean(axis=0) for cls in self.classes_]
        )

        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['means_', 'classes_'])

        # Validate the input array
        X = check_array(X)

        # Compute the distance to each class mean
        distances = np.array(
            [np.linalg.norm(X - mean, axis=1) for mean in self.means_]
        )

        # Predict the class with the closest mean
        predictions = self.classes_[np.argmin(distances, axis=0)]

        return predictions

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['means_', 'classes_'])

        # Validate the input array
        X = check_array(X)

        # Compute the distance to each class mean
        distances = np.array(
            [np.linalg.norm(X - mean, axis=1) for mean in self.means_]
        )

        # Convert distances to probabilities (inverse distance)
        proba = 1 / (distances + 1e-10)
        proba = proba / proba.sum(axis=0)

        return proba.T


############################################
# Use the custom estimator on IRIS dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

############################################
# Load the dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and fit the custom classifier
clf = CustomClassifier(threshold=0.5)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
############################################
