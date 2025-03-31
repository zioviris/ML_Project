import sqlite3
import pandas as pd
from config import DATABASE_PATH
from logging_config import setup_logger

# Setup logger
logger = setup_logger()

class Database:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        logger.info(f"Database initialized with path: {self.db_path}")

    def fetch_data(self, query):
        """Fetch data from SQLite database with error handling and logging."""
        try:
            logger.info(f"Fetching data with query: {query}")
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn)
            logger.info(f"Fetched {len(df)} rows successfully.")
            return df
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
