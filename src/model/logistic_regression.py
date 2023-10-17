import numpy as np
from numpy import ndarray
from src.classifier_model import Classifier
import logging

class LogisticRegression(Classifier):

    """
    Logistic Regression classifier implementation.

    Attributes:
        lr (float): The learning rate for gradient descent.
        n_iters (int): The number of iterations for gradient descent.
        weights (ndarray): Model weights.
        bias (float): Model bias.

    Methods:
        _sigmoid(z): Calculate the sigmoid of a given value.
        fit(X, y): Fit the Logistic Regression model to the training data.
        predict(X): Make predictions using the Logistic Regression model.
    """

    def __init__(self, lr: float =0.01, n_iters: int =100):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    @staticmethod
    def _sigmoid(z: ndarray) -> ndarray:

        """
        Initialize the Logistic Regression model.

        Args:
            lr (float): The learning rate for gradient descent.
            n_iters (int): The number of iterations for gradient descent.

        Returns:
            None
        """

        # avoid overflow problems
        z = np.float128(z)
        return 1 / (1 + np.exp(-z))


    def fit(self, X: np.ndarray, y: np.ndarray)-> None:

        """
        Fit the Logistic Regression model to the training data.

        Parameters:
        ----------
            X (ndarray): The input features for training.
            y (ndarray): The target labels for training.

        Returns:
        ---------
        None
        """

        try:
            logging.info("Fitting the Logistic Regression model.")
            n_samples: int
            n_features: int
            n_samples, n_features = X.shape

            self.weights = np.random.normal(0,0.01, n_features)
            self.bias = 0

            for _ in range(self.n_iters):
                linear_pred: ndarray = np.dot(X, self.weights) + self.bias
                y_pred: ndarray = self._sigmoid(linear_pred)

                dw: ndarray = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db: ndarray = (1 / n_samples) * np.sum(y_pred - y)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

        except Exception as e:
            logging.error(f"Error fitting the Logistic Regression model: {e}")

        finally:
            logging.info("Logistic Regression model fitting completed successfully.")


    def predict(self, X: ndarray)-> ndarray:

        """
        Make predictions using the Logistic Regression model.

        Parameters:
        -----------
            X (ndarray): The input features for making predictions.

        Returns:
        ----------
            ndarray: The predicted labels.
        """

        try:
            logging.info("Making predictions with the Logistic Regression model.")

            linear_pred: ndarray = X.dot(self.weights) + self.bias
            y_pred: ndarray = self._sigmoid(linear_pred)
            prob: ndarray = np.array([0 if y <= 0.5 else 1 for y in y_pred])
            return prob
        except Exception as e:
            logging.error(f"Error making predictions with the Logistic Regression model: {e}")
            return None
        finally:
            logging.info("Logistic Regression model prediction completed successfully.")










