import pandas as pd
import psycopg2 as ps
from psycopg2.extensions import connection 
from psycopg2.extensions import cursor
from typing import Dict, List, Tuple
import logging


class Database:

    """
    Database class for handling database connections and queries.

    Args:
    ---------
        components (Dict[str, str]): A dictionary containing database connection components.
            - 'DB_NAME': Name of the database.
            - 'USER': Username for database access.
            - 'PASSWORD': Password for database access.
            - 'HOST': Database host.
            - 'PORT': Database port.

    Attributes:
    -----------
        connection (connection): The database connection object.
        cursor (cursor): The database cursor object.
    """

    def __init__(self, components: Dict[str, str]) -> None:

        """
        Initialize a Database instance.

        Args:
        ---------
            components (Dict[str, str]): A dictionary containing database connection components.
        """

        self.connection: connection = Database._connect(components)
        self.cursor: cursor = self.connection.cursor()

    @staticmethod
    def _connect(components: Dict[str, str])-> connection:

        """
        Establish a database connection.

        Parameters:
        -----------
            components (Dict[str, str]): A dictionary containing database connection components.

        Returns:
        ----------
            connection: A database connection object.
        """

        try:
            logging.info("Trying to reach connection to database.")
            return ps.connect(dbname = components['DB_NAME'],
                              user = components['USER'],
                              password = components['PASSWORD'],
                              host = components['HOST'],
                              port = components['PORT'])

        except Exception as e:
            logging.error(f"Error: {e}")

    def execute_query(self, query: str)-> None:

        """
        Execute a SQL query.

        Parameters:
        ------------
            query (str): The SQL query to execute.

        Returns:
        ----------
            None
        """
        
        try:
           
           self.cursor.execute(query)
           logging.info("Query executed successfully")
        except Exception as e:
            logging.error(f"Error executing query: {e}")

    def _get_cursor(self)-> List:

        """
        Get the cursor's fetchall result as a list.

        Returns:
        ---------
            List: A list containing the fetchall result.
        """

        return list(self.cursor.fetchall())

    def fetch_as_dataframe(self)-> pd.DataFrame:

        """
        Fetch data from the cursor and return it as a Pandas DataFrame.

        Returns:
        ---------
            pd.DataFrame: A Pandas DataFrame containing the fetched data.
        """

        try:
            #data: List[Tuple] = self.cursor.fetchall()
            columns: List[str] = [desc[0] for desc in self.cursor.description]
            data: List =  self._get_cursor()
            return pd.DataFrame(data=data, columns = columns)
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return None

        





    




