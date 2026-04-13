# database_manager.py

import logging
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import traceback
import sqlalchemy as sa
import json
from config import get_db_url, USE_SQLITE

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
                if not database_exists(self.engine.url):
                    create_database(self.engine.url)
                    logger.info("Database created successfully.")
            
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
        """Safely clears all table data."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM argo_profiles;"))
                logger.info("Argo profiles table has been cleared.")

                conn.execute(text("DELETE FROM float_metadata;"))
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

        # ---- FIX 1: Normalize column names to match database schema ----
        column_mapping = {
            "lat": "latitude",
            "lon": "longitude"
        }

        df = df.rename(columns=column_mapping)

        # ---- FIX 2: Remove geom column if DB doesn't have it ----
        if "geom" in df.columns:
            df = df.drop(columns=["geom"])

        # ---- FIX 3: Convert metadata values safely ----
        metadata_str = {}
        for key, value in metadata.items():

            if isinstance(value, bytes):
                metadata_str[key] = value.decode('utf-8').strip()

            elif isinstance(value, list):
                metadata_str[key] = json.dumps(value)

            else:
                metadata_str[key] = value

        float_id_str = metadata_str['float_id']

        try:
            with self.engine.connect() as conn:

                # ---- Insert metadata with UPSERT ----
                stmt = text("""
                    INSERT INTO float_metadata
                    (float_id, wmo_id, project_name, institution, date_launched, parameters)
                    VALUES
                    (:float_id, :wmo_id, :project_name, :institution, :date_launched, :parameters)
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

                # ---- Remove old data for that float ----
                conn.execute(
                    text("DELETE FROM argo_profiles WHERE float_id = :float_id"),
                    {'float_id': float_id_str}
                )
                conn.commit()

                # Remove rows with missing values
                df = df.dropna(subset=["temperature", "salinity", "depth"])
                df = df.reset_index(drop=True)

                logger.info(f"Cleaned dataframe size: {len(df)}")

                df.to_sql(
                    'argo_profiles',
                    self.engine,
                    if_exists='append',
                    index=False,
                    chunksize=1000,
                    method="multi"
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