# Earthquake_Damage_Predictions_Modularized

In the [Traditional ML](https://github.com/GSana2812/Traditional_ML)  repository, among other projects, one notable project was predicting whether a building suffered severe damage or not during the catastrophic earthquake in Nepal, 2015. This project was originally done in a jupyter notebookk file, and I thought about structuring it a little bit, using Object Oriented Programming. This could help me further improve my software engineering skills combined with predictive analytics. 

The original dataset, is an sqlite file, that after converting it into a .sql file readable by postgresql, have extracted the necessary data by connecting postgresql and python. Then after performing some data preprocessing, I have developed the logistic regression model from scratch using numpy. There are also 2 other models, such as Decision Trees and Random Forest, which are taken from the sklearn library. My intention has been to compare these 3 models, and by far Random Forest and Decision Trees hold the first place, but with Logistic Regression being not far ahead, with an accuracy of 75..%. I also created a class called Tuning, which helped me decide the best hyperparameters to choose, using GridSearchCV of sklearn. For more, feel free to look around the notebooks. They have all the necessary results.

Below there are mentioned the classes created (for clarity) and libraries used:
1. Database (connection to database)
2. DataPreprocessor (data cleaning and preprocessing)
3. Tuning (choosing the best hyperparameters)
4. Classifier (an abstract class which will serve as the main superclass for Logistic Regression and potential models that can be developed from scratch in the future)
5. Logistic Regression (implemented from scratch including updating weights using gradient descent algorithm)

Libraries: sklearn, pandas, numpy, matplotlib, psycopg2, toml, logging
