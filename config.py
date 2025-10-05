# config.py

# Set this to False to use PostgreSQL
USE_SQLITE = False 

# PostgreSQL database URL
# Replace with your actual credentials if they are different
POSTGRES_DB_URL = "postgresql://postgres:123123@localhost:5432/argo_db" 

# Data processing configuration
DATA_PROCESSING_CONFIG = {
    "data_dir": "data/",
    "max_files": None  # Process all files
}

# Vector database configuration
VECTOR_DB_CONFIG = {
    "path": "C:\\Users\\ACER\\Documents\\Float-Chat\\chroma_db",  # Updated to absolute path
    "collection_name": "argo_metadata",
    "embedding_model": "all-MiniLM-L6-v2"
}

# Ollama LLM configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "llama3:8b", # Or your preferred model
    "temperature": 0.1
}

def get_db_url():
    return "sqlite:///argo_db.db" if USE_SQLITE else POSTGRES_DB_URL

def get_ollama_model():
    return OLLAMA_CONFIG["model"]