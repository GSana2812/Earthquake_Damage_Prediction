import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import OneHotEncoder
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
import logging

class DataPreprocessor:

    @staticmethod
    def drop_high_leakage_features(data: pd.DataFrame, columns: List[str])-> None:

        """
        Drop columns with high leakage indicators in the provided DataFrame.

        Parameters:
        ----------
            data (pd.DataFrame): The DataFrame containing the data.
            columns (List[str]): A list of column names to identify high leakage columns.

        Returns:
        ----------
            None: Modifies the input DataFrame in place by dropping the specified columns.
        """
        try:
            logging.info("Removing High Leakage features")
            leakage_cols: List[str] = [leakage for leakage in columns if 'post_eq' in leakage]
            data.drop(columns = leakage_cols, inplace = True)
        
        except Exception as e:
            logging.error(f"Error in removing high leakage features: {str(e)}")
        finally:
            logging.info("High Leakage features removed successfully")



    @staticmethod
    def drop_multi_collinearity_columns(data: pd.DataFrame, column: str)-> None:

        """
        Drop a column to address multicollinearity in the provided DataFrame.

        Parameters:
        ----------
            data (pd.DataFrame): The DataFrame containing the data.
            column (str): The column name to drop.

        Returns
        --------
            None: Modifies the input DataFrame in place by dropping the specified column.
        """

        try:
            logging.info("Removing columns with high correlation!")
            data.drop(columns = [column], inplace=True)
        except Exception as e:
            logging.error(f"Error in removing high correlated features : {str(e)}")
        finally:
            logging.info("Highly correlated columns successfully removed!")

    @staticmethod
    def drop_not_significant_values(data: pd.DataFrame, column: str)-> None:

        """
        Drop a column containing not so many significant values.

        Parameters:
        ----------
            data (pd.DataFrame): The DataFrame containing the data.
            column (str): The column name to drop.

        Returns
        --------
            None: Modifies the input DataFrame in place by dropping the specified column.
        """

        try:
            logging.info("Removing columns with low importance!")
            data.drop(columns = [column], inplace=True)
        except Exception as e:
            logging.error(f"Error in removing feature : {str(e)}")
        finally:
            logging.info("Not significant columns successfully removed!")

    @staticmethod
    def drop_low_high_cardinality_features(data: pd.DataFrame, column: List[str])-> None:

        # high cardinality refers to caategorical variables that have a lot of distinct values

        """
        Drop a column with high cardinality (many distinct values) in the provided DataFrame.

        Parameters:
        ----------
            data (pd.DataFrame): The DataFrame containing the data.
            column (str): The column name to drop.

        Returns
        ---------
            None: Modifies the input DataFrame in place by dropping the specified column.
        """
        try:
            logging.info("Removing features with low or high cardinality!")
            data.drop(columns = [column], inplace=True)
        except Exception as e:
            logging.error(f"Error in removing high or low cardinality features: {str(e)}")
        finally:
            logging.info("High or low cardinality features successfully removed.")

    @staticmethod
    def create_severe_damage(data: pd.DataFrame, column: str)-> None:
        """
        Create a new column 'severe_damage' based on a condition in the provided DataFrame.

        Parameters:
        ----------
            data (pd.DataFrame): The DataFrame containing the data.
            column (str): The column name to use for creating the 'severe_damage' column.

        Returns:
        ---------
            None: Modifies the input DataFrame in place by adding the 'severe_damage' column.
        """
        try:
            logging.info("Creating the target variable: severe damage")
            data['severe_damage']: pd.Series = (data['damage'] > 3).astype(int)
        except Exception as e:
            logging.error(f"Error in creating severe damage: {str(e)}")

    @staticmethod
    def drop_damage_grade_column(data: pd.DataFrame, column: str)-> None:
        """
        Drop the 'damage' column in the provided DataFrame.

        Parameters:
        ----------
            data (pd.DataFrame): The DataFrame containing the data.
            column (str): The column name to drop.

        Returns:
        ---------
            None: Modifies the input DataFrame in place by dropping the 'damage' column.
        """
        try:
            logging.info("Removing the damage grade column")
            data.drop(columns = [column], inplace=True)
        except Exception as e:
            logging.error(f"Error in removing damage grade column: {str(e)}")
        finally:
            logging.info("Damage grade column successfully removed.")

    @staticmethod
    def one_hot_encoder(data: pd.DataFrame, columns_to_encode:List[str])-> pd.DataFrame:
        """
        Apply one-hot encoding to specified columns in the provided DataFrame.

        Parameters:
        ----------
            data (pd.DataFrame): The DataFrame containing the data.
            columns_to_encode (List[str]): A list of column names to one-hot encode.

        Returns:
        ---------
            pd.DataFrame: A new DataFrame with the specified columns one-hot encoded and concatenated.
        """
        try:
            logging.info("Encoding categorical variables")
            encoder = OneHotEncoder(sparse=False)
            encoded_data: ndarray = encoder.fit_transform(data[columns_to_encode])

            # we must specify the updated column names as actual columns of the encoded dataframe
            encoded_data_df = pd.DataFrame(encoded_data, columns = encoder.get_feature_names_out())


            #drop these columns from main dataframe to concat them after
            data = data.drop(columns = columns_to_encode)

            return pd.concat([data, encoded_data_df], axis=1)
        except Exception as e:
            logging.error(f"Error in encoding categorical values: {str(e)}")
        finally:
            logging.info("Categorical values successfully encoded!")

    @staticmethod
    def feature_scaler(data: pd.DataFrame, columns_to_scale: List[str])-> pd.DataFrame:

        """
        Apply feature scaling on the features.

        Parameters
        ----------
        - scaler : StandardScaler
        The scaler from scikit-learn library
        - data : pd.DataFrame

        Returns
        -------
        pd.DataFrame:
        The dataframe with its features scaled.
        """
        try:
            logging.info("Applying Feature Scaling on dataframe columns")
            scaler = StandardScaler()

            scaled_df = pd.DataFrame(scaler.fit_transform(data[columns_to_scale]), columns=columns_to_scale)
            data = data.drop(columns = columns_to_scale)

            return pd.concat([data, scaled_df], axis=1)
        except Exception as e:
            logging.error(f"Error in applying feature scaling {e}")
        finally:
            logging.info("Successfully scaled features")




