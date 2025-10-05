import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import logging
import traceback

from database_manager import DatabaseManager
from rag_system import VectorStoreManager
from config import DATA_PROCESSING_CONFIG, USE_SQLITE

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class NcToPostgresProcessor:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_PROCESSING_CONFIG["data_dir"]
        self.db_manager = DatabaseManager()
        self.vector_store = VectorStoreManager()

    def _decode_if_bytes(self, value):
        """
        Decodes a byte string, a numpy byte array, or a numpy array 
        with a single value into a clean string.
        """
        try:
            # Handle NumPy byte arrays, which are common in xarray datasets
            if isinstance(value, np.bytes_):
                return value.decode('utf-8').strip()
            # Handle standard Python byte strings
            elif isinstance(value, bytes):
                return value.decode('utf-8').strip()
            # If it's a NumPy array, get the single item and process it
            elif isinstance(value, np.ndarray) and value.size == 1:
                return self._decode_if_bytes(value.item())
            # If it's a regular string, just strip whitespace
            elif isinstance(value, str):
                return value.strip()
            # For any other data type, convert to string
            else:
                return str(value).strip()
        except Exception as e:
            logger.error(f"Error decoding value: {value}, Error: {e}")
            return str(value).strip()

    def _extract_metadata(self, ds):
        try:
            first_prof = ds.isel(N_PROF=0)
            
            metadata = {
                "float_id": self._decode_if_bytes(first_prof['PLATFORM_NUMBER'].values),
                "wmo_id": self._decode_if_bytes(first_prof['WMO_INST_TYPE'].values),
                "project_name": self._decode_if_bytes(first_prof['PROJECT_NAME'].values),
                "institution": self._decode_if_bytes(ds.attrs.get('institution', b'Unknown')),
                "date_launched": str(self._decode_if_bytes(ds.attrs.get('date_update', 'N/A'))),
                "parameters": [var for var in ds.data_vars if '_ADJUSTED' in var]
            }
            return metadata
        except Exception as e:
            logger.error(f"Could not extract metadata: {e}")
            return None

    def _generate_metadata_summary(self, metadata):
        if not metadata: return None
        summary = (
            f"ARGO float ID {metadata['float_id']} from the {metadata['project_name']} project, "
            f"managed by {metadata['institution']}. It measures parameters including: "
            f"{', '.join(metadata['parameters'])}."
        )
        return summary

    def process_netcdf_file(self, file_path):
        try:
            with xr.open_dataset(file_path, decode_times=True) as ds:
                records = []
                num_profiles = ds.sizes['N_PROF']
                num_levels = ds.sizes['N_LEVELS']

                for i in range(num_profiles):
                    try:
                        profile_data = ds.isel(N_PROF=i)
                        profile_time = profile_data['JULD'].values
                        if pd.isna(profile_time):
                            logger.warning(f"Skipping profile {i} in {os.path.basename(file_path)} due to invalid time.")
                            continue

                        lat_val = float(profile_data['LATITUDE'].values)
                        lon_val = float(profile_data['LONGITUDE'].values)
                        cycle_num = int(profile_data['CYCLE_NUMBER'].values)
                        float_id = self._decode_if_bytes(profile_data['PLATFORM_NUMBER'].values)

                        for j in range(num_levels):
                            try:
                                pres_val = profile_data['PRES_ADJUSTED'].values[j]
                                temp_val = profile_data['TEMP_ADJUSTED'].values[j]
                                sal_val = profile_data['PSAL_ADJUSTED'].values[j]
                                
                                if not np.isnan(pres_val):
                                    records.append({
                                        'float_id': float_id,
                                        'profile_number': cycle_num,
                                        'time': pd.to_datetime(profile_time),
                                        'lat': lat_val,
                                        'lon': lon_val,
                                        'depth': float(pres_val),
                                        'temperature': float(temp_val) if not np.isnan(temp_val) else None,
                                        'salinity': float(sal_val) if not np.isnan(sal_val) else None
                                    })
                            except Exception as e:
                                logger.error(f"Error processing level {j} in profile {i}: {e}")

                    except Exception as e:
                        logger.error(f"Error processing profile {i} in {os.path.basename(file_path)}: {e}")
                
                if not records:
                    logger.warning(f"No valid records found in file: {os.path.basename(file_path)}")
                    return None, None
                
                df = pd.DataFrame(records)
                metadata = self._extract_metadata(ds)
                
                return df, metadata
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None, None

    def process_directory(self, max_files=None):
        self.db_manager.initialize_database()
        
        self.db_manager.clear_all_data()
        self.vector_store.clear_collection()
        logger.info("Database tables and vector store have been cleared. Starting with a fresh import.")
        
        nc_files = sorted(glob.glob(os.path.join(self.data_dir, "*.nc")))
        if max_files: nc_files = nc_files[:max_files]
        
        logger.info(f"Found {len(nc_files)} NetCDF files to process.")
        
        for file_path in nc_files:
            logger.info(f"Processing file: {os.path.basename(file_path)}")
            df, metadata = self.process_netcdf_file(file_path)
            
            if df is not None and not df.empty and metadata:
                logger.info(f"Extracted {len(df)} measurements from {os.path.basename(file_path)}")
                
                if not USE_SQLITE:
                    df['geom'] = df.apply(lambda row: f'SRID=4326;POINT({row.lon} {row.lat})', axis=1)

                success = self.db_manager.insert_argo_data(df, metadata)
                
                if success:
                    logger.info(f"Successfully inserted data for float {metadata['float_id']} into PostgreSQL.")
                    
                    summary_text = self._generate_metadata_summary(metadata)
                    self.vector_store.add_document(
                        doc_id=metadata['float_id'],
                        document=summary_text,
                        metadata={"float_id": metadata['float_id']}
                    )
                    logger.info(f"Successfully added metadata for float {metadata['float_id']} to ChromaDB.")
                else:
                    logger.error(f"Failed to insert data for float {metadata['float_id']}.")
            else:
                logger.warning(f"No valid data found in file: {os.path.basename(file_path)}. Skipping.")

if __name__ == "__main__":
    processor = NcToPostgresProcessor()
    processor.process_directory(max_files=DATA_PROCESSING_CONFIG["max_files"])
    print("✅ NetCDF to PostgreSQL and ChromaDB processing finished.")