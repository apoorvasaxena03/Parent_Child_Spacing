# %%
import pyodbc
import pandas as pd
from src.custom_logger import CustomLogger

#%%
# Create an instance of CustomLogger with logger name and log directory
# logger_instance = CustomLogger("scraper","DB_Manager", r"C:\Users\Apoorva.Saxena\OneDrive - Sitio Royalties\Desktop\Project - Apoorva\Python\Parent_Child_Spacing\src\logs")

# Get the logger
# db_logger = logger_instance.get_logger()
#%%
class SQLTableManager:
    def __init__(self) -> None:
        self.driver:str = '{SQL Server}'
        self.server:str = 'TXDC-LHACNDB01'
        self.trusted_connection:str = 'yes'
        self.connection = None


    def connect(self, dbname:str = 'master') -> None:
        # Establish a connection
        self.connection = pyodbc.connect(
            driver = self.driver,
            server = self.server,
            database = dbname,
            trusted_connection = self.trusted_connection
        )
        self.cursor = self.connection.cursor()


    def execute_query(self, sql_query:str) -> pd.DataFrame:
        if self.connection is None:
            raise Exception("Not connected to SQL Server. Call connect() first.")
        
        # Execute the query and fetch the result into a Pandas DataFrame
        result_df = pd.read_sql(sql_query, self.connection)
        
        return result_df


    # def add_rows_from_dataframe(self, sql_table_name:str, dataframe:pd.DataFrame) -> None:
        
    #     # Convert column names to tuple without apostrophes
    #     column_tuple = tuple(dataframe.columns)
        
    #     # Remove apostrophes from the string representation of the tuple
    #     column_tuple_str = str(column_tuple).replace("'", "")

    #     for index, row in dataframe.iterrows():

    #         row_values = row.apply(lambda x: x if pd.notnull(x) else None)
    #         placeholders = ', '.join(['?'] * len(row_values))
    #         query = f"INSERT INTO {sql_table_name} {column_tuple_str} VALUES ({placeholders});"
    #         try:    
    #             self.cursor.execute(query, tuple(row_values))
    #             self.connection.commit()
    #         except Exception as e:
    #             db_logger.error(f"Error adding row to {sql_table_name}: {tuple(row_values)}.")

    def close_connection(self) -> None:
        if self.cursor:
            self.cursor.close()
        if self.connection is not None:
            self.connection.close()
            self.connection = None