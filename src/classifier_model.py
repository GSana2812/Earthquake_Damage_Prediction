from abc import ABC, abstractmethod
from numpy import ndarray
import numpy as np
from typing import Type
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

class Classifier(ABC):


    """
    Abstract base class for classifier models.

    Attributes:
        None

    Methods:
        fit(X, y): Abstract method to fit the model to the training data.
        predict(X): Abstract method to make predictions on new data.
        fit_predict(model, X_train, X_test, y): Fit the model to the training data and predict on new data.
        _compute_tp_tn_fp_fn(y_true, y_pred): Calculate true positive, true negative, false positive, and false negative.
        accuracy(y_true, y_pred): Calculate the accuracy of the predictions.
        metrics(y_true, y_pred): Calculate accuracy, precision, recall, and F1-score.
        confusion_matrix(y_true, y_pred): Display the confusion matrix.
        """

    @abstractmethod
    def fit(self, X: ndarray, y: ndarray)-> None:
        pass

    @abstractmethod
    def predict(self, X: ndarray)-> ndarray:
        pass

    @classmethod
    def fit_predict(cls, model: Type['Classifier'], X_train: ndarray, X_test: ndarray, y: ndarray)-> ndarray:

        """
        Fit the model to the training data and predict on new data.

        Parameters:
        -----------
            model (Type['Classifier']): The classifier model to use.
            X_train (ndarray): The input features for training.
            X_test (ndarray): The input features for making predictions.
            y (ndarray): The target labels for training.

        Returns:
        ----------
            ndarray: The predicted labels.
        """
        model.fit(X_train, y)
        y_pred: ndarray = model.predict(X_test)
        return y_pred

    @classmethod
    def _compute_tp_tn_fp_fn(cls, y_true: ndarray, y_pred: ndarray):
        '''
        	True positive - actual = 1, predicted = 1
        	False positive - actual = 1, predicted = 0
        	False negative - actual = 0, predicted = 1
        	True negative - actual = 0, predicted = 0
        '''
        tp: int = np.sum((y_true==1) & (y_pred == 1))
        tn: int = np.sum((y_true == 0) & (y_pred) == 0)
        fp: int = np.sum((y_true == 0)& (y_pred == 1))
        fn: int = np.sum((y_true == 1)&(y_pred == 0))

        return tp, tn, fp, fn


    @classmethod
    def metrics(cls, y_true: ndarray, y_pred: ndarray):

        """
        Calculate accuracy, precision, recall, and F1-score.

        Parameters:
        -----------
        y_true (ndarray): The true labels.
        y_pred (ndarray): The predicted labels.

        Returns:
        ---------
        Dict[str, float]: A dictionary containing accuracy, precision, recall, and F1-score.
        """

        tp, tn, fp, fn = Classifier._compute_tp_tn_fp_fn(y_true, y_pred)

        accuracy: float =  (tp + tn) / (tp + tn + fp + fn)
        precision: float = tp / (tp+fp)
        recall: float = tp / (tp+fn)
        f1_score: float = (2 * precision * recall) / (precision + recall)


        return {"Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score}

    @classmethod
    def confusion_matrix(cls, y_true: ndarray, y_pred: ndarray):
        """
        Display a confusion matrix plot.

        Parameters:
        ----------
            y_true (ndarray): The true labels.
            y_pred (ndarray): The predicted labels.

        Returns:
        --------
            None
        """

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))

        # display the confusion matrix
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot(cmap="Blues",ax=ax,colorbar=False)
        plt.show()


