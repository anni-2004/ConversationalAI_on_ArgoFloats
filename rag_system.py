# rag_system.py

import logging
import pandas as pd
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import chromadb
from database_manager import DatabaseManager
from config import OLLAMA_CONFIG, VECTOR_DB_CONFIG, get_ollama_model
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=VECTOR_DB_CONFIG["path"])
            
            # Use get_or_create_collection for robustness
            self.collection = self.client.get_or_create_collection(
                name=VECTOR_DB_CONFIG["collection_name"]
            )
            
            try:
                # This line now correctly pulls from the config
                self.embedding_model = SentenceTransformer(VECTOR_DB_CONFIG["embedding_model"])
            except Exception as e:
                logger.error(f"Could not load SentenceTransformer model: {e}")
                self.embedding_model = None
        except Exception as e:
            logger.error(f"FATAL ERROR: Could not connect to or initialize ChromaDB. Details: {e}")
            raise

    def clear_collection(self):
        """Deletes and recreates the collection to clear all data."""
        try:
            self.client.delete_collection(name=self.collection.name)
            logger.info(f"Successfully deleted ChromaDB collection: {self.collection.name}")
            self.collection = self.client.create_collection(name=self.collection.name)
            logger.info(f"Recreated empty ChromaDB collection: {self.collection.name}")
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collection: {e}")
            raise
    
    def add_document(self, doc_id: str, document: str, metadata: dict):
        if not self.embedding_model:
            logger.warning("Embedding model not loaded. Skipping add_document.")
            return
        try:
            embedding = self.embedding_model.encode(document).tolist()
            self.collection.upsert(
                documents=[document],
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"ChromaDB - Error adding document {doc_id}: {e}")

    def search(self, query: str, n_results: int = 3):
        if not self.embedding_model:
            logger.warning("Embedding model not loaded. Skipping search.")
            return []
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results.get('documents')[0] if results.get('documents') else []
        except Exception as e:
            logger.error(f"ChromaDB - Error searching: {e}")
            return []

class RAGSQLQueryExecutor:
    def __init__(self, ollama_model=None):
        self.db_manager = DatabaseManager()
        self.vector_store = VectorStoreManager()
        try:
            self.llm = Ollama(
                base_url=OLLAMA_CONFIG["base_url"],
                model=ollama_model or get_ollama_model(),
                temperature=OLLAMA_CONFIG["temperature"]
            )
            logger.info(f"✅ Connected to Ollama model: {self.llm.model}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            raise

        self.sql_prompt_template = """You are an expert oceanographer and SQL developer working with an ARGO float PostgreSQL database. Your task is to generate a single, correct SQL query that answers the user's question.

### RULES:
1. ALWAYS generate a simple PostgreSQL query using the table `argo_profiles`.
2. Use simple SQL with `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, and aggregation functions when necessary.
3. DO NOT use complex subqueries or `JOIN`s.
4. The available columns are:
   - float_id
   - profile_number
   - time
   - latitude
   - longitude
   - depth
   - temperature
   - salinity
5. For filtering by float, use the `float_id` column.
6. For filtering by time or date, use the `time` column.
7. For time-of-day filtering (e.g., "night", "morning"), cast timestamp using `time::time`.
8. For depth filtering, use the `depth` column.
9. For temperature analysis, use the `temperature` column.
10. For salinity analysis, use the `salinity` column.
11. Return ONLY the SQL query.
12. Do NOT include explanations, markdown, or comments.
13. End the SQL query with a semicolon.

### EXAMPLES:

User Question: How many unique floats are there?
SQL Query:
SELECT COUNT(DISTINCT float_id) FROM argo_profiles;

User Question: Show temperature depth profile for float 5906256.
SQL Query:
SELECT depth, temperature FROM argo_profiles
WHERE float_id = '5906256'
ORDER BY depth ASC;

User Question: What is the average temperature at 1000 meters?
SQL Query:
SELECT AVG(temperature)
FROM argo_profiles
WHERE depth BETWEEN 950 AND 1050;

User Question: Show the latest 10 measurements.
SQL Query:
SELECT *
FROM argo_profiles
ORDER BY time DESC
LIMIT 10;

User Question: What is the minimum temperature recorded?
SQL Query:
SELECT MIN(temperature)
FROM argo_profiles;

User Question: Which float recorded the highest salinity?
SQL Query:
SELECT float_id, MAX(salinity)
FROM argo_profiles
GROUP BY float_id
ORDER BY MAX(salinity) DESC
LIMIT 1;

---

### DATABASE SCHEMA:
Table: argo_profiles

Columns:
float_id (text)
profile_number (integer)
time (timestamp)
latitude (double precision)
longitude (double precision)
depth (double precision)
temperature (double precision)
salinity (double precision)

---

### CONTEXT FROM DATA METADATA:
{context}

### CURRENT USER QUESTION:
{question}

SQL Query:
"""
        self.response_prompt_template = """You are an expert oceanographer acting as a helpful AI assistant.

### USER QUESTION
{question}

### DATA SUMMARY
{data_summary}

### INSTRUCTIONS
1. Briefly explain the result in simple language.
2. Mention any oceanographic insight if possible.
3. Keep the answer short and clear.

Response:
"""
    def _generate_sql(self, question: str, context: str):
        prompt = self.sql_prompt_template.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        sql_query = response.strip().replace("```sql", "").replace("```", "").replace(";", "") + ";"
        return sql_query
    
    def _summarize_results(self, question: str, df: pd.DataFrame):
        if df.empty:
            return "The query returned no results. This could mean there is no data matching your criteria."
        
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            col_name = df.columns[0]
            if isinstance(value, float):
                value_str = f"{value:,.2f}"
            else:
                value_str = str(value)
            data_summary = f"The query returned a single value for '{col_name}': {value_str}."
        else:
            data_summary = f"The query returned {len(df)} data points. Here is a sample:\n{df.head().to_string()}"
            
        prompt = self.response_prompt_template.format(question=question, data_summary=data_summary)
        return self.llm.invoke(prompt)

    def query_with_rag(self, user_question: str):
        try:
            logger.info(f"Searching for context related to: '{user_question}'")
            context_docs = self.vector_store.search(user_question)
            context = "\n".join(f"- {doc}" for doc in context_docs) if context_docs else "No specific context found."
            
            logger.info(f"Generating SQL with context:\n{context}")
            sql_query = self._generate_sql(user_question, context)
            logger.info(f"Generated SQL: {sql_query}")

            query_result = self.db_manager.execute_query(sql_query)
            
            if not query_result["success"]:
                raise Exception(query_result["error"])

            df = pd.DataFrame(query_result['data'], columns=query_result['columns'])
            enhanced_response = self._summarize_results(user_question, df)
            
            return {
                "success": True,
                "enhanced_response": enhanced_response,
                "generated_query": sql_query,
                "data": query_result['data'],
                "columns": query_result['columns']
            }
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "enhanced_response": f"I'm sorry, an error occurred: {e}",
                "data": [],
                "columns": []
            }