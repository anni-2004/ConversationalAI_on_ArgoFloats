# database_manager.py

import logging
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import traceback
import sqlalchemy as sa
import json  # Added to handle JSON data
from config import get_db_url, USE_SQLITE # ADD THIS LINE

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.db_url = get_db_url()
        self.engine = create_engine(self.db_url)
        logger.info(f"Connecting to database at: {self.db_url}")

    def initialize_database(self):
        try:
            if not USE_SQLITE:
                # Check for and create the database
                if not database_exists(self.engine.url):
                    create_database(self.engine.url)
                    logger.info("Database created successfully.")
            
            # Create tables from a known good schema
            schema_file = 'database_schema.sql'
            if not USE_SQLITE and self.engine.dialect.has_table(self.engine.connect(), 'argo_profiles'):
                logger.info("Tables already exist. Skipping table creation.")
                return

            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            with self.engine.connect() as conn:
                conn.execute(text(schema_sql))
                conn.commit()
            
            logger.info("Database schema created successfully.")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def clear_all_data(self):
        """Truncates all data from the database tables using raw SQL."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("TRUNCATE TABLE argo_profiles RESTART IDENTITY;"))
                logger.info("Argo profiles table has been cleared.")
                conn.execute(text("TRUNCATE TABLE float_metadata RESTART IDENTITY;"))
                logger.info("Float metadata table has been cleared.")
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error clearing tables: {e}")
            return False

    def insert_argo_data(self, df: pd.DataFrame, metadata: dict):
        if df.empty:
            logger.warning("DataFrame is empty, skipping insertion.")
            return True

        # Convert bytes to string and lists to JSON for consistency
        metadata_str = {}
        for key, value in metadata.items():
            if isinstance(value, bytes):
                metadata_str[key] = value.decode('utf-8').strip()
            elif isinstance(value, list):
                # Correctly convert the Python list to a JSON string
                metadata_str[key] = json.dumps(value)
            else:
                metadata_str[key] = value
        
        float_id_str = metadata_str['float_id']

        try:
            with self.engine.connect() as conn:
                # Insert metadata with a simple UPSERT
                stmt = text("""
                    INSERT INTO float_metadata (float_id, wmo_id, project_name, institution, date_launched, parameters)
                    VALUES (:float_id, :wmo_id, :project_name, :institution, :date_launched, :parameters)
                    ON CONFLICT (float_id) DO UPDATE SET
                        wmo_id = EXCLUDED.wmo_id,
                        project_name = EXCLUDED.project_name,
                        institution = EXCLUDED.institution,
                        date_launched = EXCLUDED.date_launched,
                        parameters = EXCLUDED.parameters;
                """)
                conn.execute(stmt, metadata_str)
                conn.commit()
                logger.info(f"Inserted metadata for float {float_id_str}.")

                # Clean up old data for this float before inserting new data
                conn.execute(text("DELETE FROM argo_profiles WHERE float_id = :float_id"), {'float_id': float_id_str})
                conn.commit()

                # Use pandas to_sql for efficient bulk insert
                df.to_sql(
                    'argo_profiles', 
                    self.engine, 
                    if_exists='append', 
                    index=False
                )
                conn.commit()
                
                logger.info(f"Inserted {len(df)} rows for float {float_id_str}.")
            return True

        except Exception as e:
            logger.error(f"Error inserting data for float {float_id_str}: {e}")
            logger.error(traceback.format_exc())
            return False

    def execute_query(self, query: str):
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(text(query), connection)
            return {
                "success": True,
                "data": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "rowcount": len(df)
            }
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {"success": False, "error": str(e), "data": []}

    def get_metrics(self):
        try:
            with self.engine.connect() as conn:
                active_floats_query = "SELECT COUNT(DISTINCT float_id) FROM float_metadata;"
                active_floats = conn.execute(text(active_floats_query)).scalar()

                ocean_profiles_query = "SELECT COUNT(*) FROM argo_profiles;"
                ocean_profiles = conn.execute(text(ocean_profiles_query)).scalar()

                data_points_query = "SELECT COUNT(*) FROM argo_profiles;"
                data_points = conn.execute(text(data_points_query)).scalar()

            return {
                "success": True,
                "active_floats": active_floats,
                "ocean_profiles": ocean_profiles,
                "data_points": data_points
            }
        except Exception as e:
            logging.error(f"Error fetching metrics: {e}")
            return {
                "success": False,
                "error": str(e)
            }