import numpy as np
from numpy import ndarray
from sklearn.model_selection import GridSearchCV
from src.classifier_model import Classifier
from typing import Dict, Tuple

class Tuning:

    def __init__(self, model: Classifier, param_grid: Dict[str,str]):
        """
        Initialize the Tuning class with a machine learning model and a hyperparameter grid.

        Parameters
        ----------
        model (Classifier): The machine learning model to be tuned.
        param_grid (Dict[str, str]): A dictionary specifying the hyperparameters and their possible values.

        Returns
        ---------
        None
        """

        self.model = model
        self.param_grid = param_grid

    def tune(self, X: ndarray, y: ndarray)-> Tuple[float, float]:
        """
        Perform hyperparameter tuning for the specified model.

        Parameters
        ----------
        X (ndarray): The input features for training.
        y (ndarray): The target labels for training.

        Returns
        ---------
        Tuple[float, float]: A tuple containing the best hyperparameters and their corresponding best score.
        """

        grid_search = GridSearchCV(self.model, self.param_grid, cv = 5, scoring = 'accuracy')
        grid_search.fit(X,y)

        best_params: float = grid_search.best_params_
        best_score: float = grid_search.best_score_

        return best_params, best_score