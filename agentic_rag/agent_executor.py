import sys
import os
import json
import logging
import pandas as pd
from datetime import datetime

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSQLQueryExecutor
from database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgenticRAG")

class AgenticSQLQueryExecutor:
    def __init__(self):
        self.rag_executor = RAGSQLQueryExecutor()
        self.db_manager = DatabaseManager()
        self.logs_dir = "agentic_rag/logs"
        os.makedirs(self.logs_dir, exist_ok=True)

    def _get_correction_prompt(self, question, failed_sql, error_msg):
        # Extract schema part from the prompt template
        schema_info = self.rag_executor.sql_prompt_template.split("### DATABASE SCHEMA:")[1].split("### CONTEXT FROM DATA METADATA:")[0]
        
        prompt = f"""
        USER QUESTION: {question}
        
        PREVIOUS ATTEMPTED SQL:
        {failed_sql}
        
        POSTGRESQL ERROR:
        {error_msg}
        
        THE PREVIOUS SQL FAILED. Please fix it based on the error message and the schema provided below.
        
        SCHEMA CONTEXT:
        {schema_info}
        
        RULES:
        1. Return ONLY the corrected SQL query.
        2. Do not explain anything.
        3. Ensure valid PostgreSQL syntax.
        4. If the error was an 'UndefinedTable', ensure you are joining argo_profiles with float_metadata correctly.
        """
        return prompt

    def query_with_rag(self, question):
        return self.query_with_correction(question)

    def query_with_correction(self, question):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_query": question,
            "attempts": []
        }
        
        # 1. Retrieve Context
        context_docs = self.rag_executor.vector_store.search(question)
        context_str = "\n".join(f"- {doc}" for doc in context_docs) if context_docs else "No specific context found."

        # 2. ATTEMPT 1: Initial SQL Generation
        logger.info(f"Attempt 1 for: {question}")
        try:
            initial_sql = self.rag_executor._generate_sql(question, context_str)
        except Exception as e:
            logger.error(f"Attempt 1 generation failed: {e}")
            log_entry["final_status"] = "Generation 1 Failed"
            self._save_log(log_entry)
            return {
                "success": False, 
                "enhanced_response": f"Failed to generate initial SQL: {e}", 
                "status": "Failed"
            }

        # 3. ATTEMPT 1: Execution
        res = self.db_manager.execute_query(initial_sql)
        attempt_1 = {
            "attempt": 1,
            "sql": initial_sql,
            "success": res["success"],
            "error": res.get("error") if not res["success"] else None
        }
        log_entry["attempts"].append(attempt_1)

        if res["success"]:
            logger.info("Attempt 1 Succeeded.")
            log_entry["final_status"] = "Success (First Pass)"
            self._save_log(log_entry)
            
            # Generate summary
            df = pd.DataFrame(res["data"])
            enhanced_response = self.rag_executor._summarize_results(question, df)
            
            return {
                "success": True,
                "enhanced_response": enhanced_response,
                "generated_query": initial_sql,
                "data": res["data"],
                "status": "Success (First Pass)"
            }

        # 4. ATTEMPT 2 (CORRECTION)
        logger.warning(f"Attempt 1 Failed: {res.get('error')}. Retrying correction...")
        correction_prompt = self._get_correction_prompt(question, initial_sql, res.get("error"))
        
        try:
            corrected_sql = self.rag_executor.llm.invoke(correction_prompt).strip()
            # Clean SQL
            if "```sql" in corrected_sql:
                corrected_sql = corrected_sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in corrected_sql:
                corrected_sql = corrected_sql.split("```")[1].strip()
        except Exception as e:
            logger.error(f"Attempt 2 generation failed: {e}")
            log_entry["final_status"] = "Generation 2 Failed"
            self._save_log(log_entry)
            return {
                "success": False, 
                "enhanced_response": f"Correction failed: {e}", 
                "status": "Failed"
            }

        # 5. ATTEMPT 2: Execution
        res_corr = self.db_manager.execute_query(corrected_sql)
        attempt_2 = {
            "attempt": 2,
            "sql": corrected_sql,
            "success": res_corr["success"],
            "error": res_corr.get("error") if not res_corr["success"] else None
        }
        log_entry["attempts"].append(attempt_2)

        if res_corr["success"]:
            logger.info("Attempt 2 (Correction) Succeeded.")
            log_entry["final_status"] = "Success (After Correction)"
            self._save_log(log_entry)
            
            # Generate summary
            df = pd.DataFrame(res_corr["data"])
            enhanced_response = self.rag_executor._summarize_results(question, df)
            
            return {
                "success": True,
                "enhanced_response": enhanced_response,
                "generated_query": corrected_sql,
                "data": res_corr["data"],
                "status": "Success (After Correction)"
            }
        else:
            logger.error("Attempt 2 (Correction) also failed.")
            log_entry["final_status"] = "Failed (Both Attempts)"
            self._save_log(log_entry)
            return {
                "success": False,
                "enhanced_response": f"Execution failed after correction: {res_corr.get('error')}",
                "generated_query": corrected_sql,
                "data": [],
                "status": "Failed"
            }

    def _save_log(self, entry):
        filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        with open(os.path.join(self.logs_dir, filename), "w") as f:
            json.dump(entry, f, indent=2)

if __name__ == "__main__":
    executor = AgenticSQLQueryExecutor()
    q = "Which institution manages float '2902121'?"
    result = executor.query_with_correction(q)
    print(json.dumps(result, indent=2))
