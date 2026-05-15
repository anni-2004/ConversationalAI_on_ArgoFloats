import json
import pandas as pd
import time
from rag_system import RAGSQLQueryExecutor, VectorStoreManager
from database_manager import DatabaseManager
import logging

# Disable verbose logging for cleaner output
logging.getLogger("rag_system").setLevel(logging.ERROR)
logging.getLogger("database_manager").setLevel(logging.ERROR)

class PipelineEvaluator:
    def __init__(self):
        self.rag_executor = RAGSQLQueryExecutor()
        self.db_manager = DatabaseManager()
        self.dataset = self.load_dataset()
        
    def load_dataset(self):
        with open('evaluation_dataset.json', 'r') as f:
            return json.load(f)

    def execute_and_compare(self, generated_sql, ground_truth_sql):
        """Executes both queries and compares the resulting dataframes."""
        try:
            gen_res = self.db_manager.execute_query(generated_sql)
            gt_res = self.db_manager.execute_query(ground_truth_sql)
            
            if not gen_res["success"]:
                return False, f"SQL Error: {gen_res.get('error')}"
            
            df_gen = pd.DataFrame(gen_res["data"])
            df_gt = pd.DataFrame(gt_res["data"])
            
            if df_gen.equals(df_gt):
                return True, "Match"
            
            # Handle cases where column names might differ but content is same
            if len(df_gen) == len(df_gt) and len(df_gen.columns) == len(df_gt.columns):
                # Sort and compare values only
                try:
                    df_gen_sorted = df_gen.sort_values(by=list(df_gen.columns)).reset_index(drop=True)
                    df_gt_sorted = df_gt.sort_values(by=list(df_gt.columns)).reset_index(drop=True)
                    if (df_gen_sorted.values == df_gt_sorted.values).all():
                        return True, "Match (Content Only)"
                except:
                    pass
            
            return False, "Data Mismatch"
        except Exception as e:
            return False, f"Comparison Error: {str(e)}"

    def evaluate_query(self, item, use_rag=True):
        question = item['question']
        gt_sql = item['ground_truth_sql']
        
        start_time = time.time()
        if use_rag:
            response = self.rag_executor.query_with_rag(question)
            generated_sql = response.get("generated_query")
            context = self.rag_executor.vector_store.search(question)
        else:
            # Non-RAG mode: Generate SQL without context
            generated_sql = self.rag_executor._generate_sql(question, context="No context provided.")
            context = []
            
        latency = time.time() - start_time
        
        if not generated_sql:
            return {
                "success": False,
                "error": "No SQL generated",
                "latency": latency
            }
            
        is_correct, status = self.execute_and_compare(generated_sql, gt_sql)
        
        # Retrieval Evaluation (Simple check)
        retrieval_score = 0
        if use_rag:
            # Check if any relevant float_id or institution from metadata is in the context
            # This is a heuristic.
            pass

        return {
            "query_id": item['id'],
            "difficulty": item['difficulty'],
            "question": question,
            "generated_sql": generated_sql,
            "ground_truth_sql": gt_sql,
            "is_correct": is_correct,
            "status": status,
            "latency": latency,
            "context_retrieved": len(context) > 0
        }

    def run_benchmark(self):
        results = {"rag": [], "non_rag": []}
        
        print(f"Starting Evaluation on {len(self.dataset)} queries...")
        
        for i, item in enumerate(self.dataset):
            print(f"[{i+1}/{len(self.dataset)}] Evaluating: {item['id']} ({item['difficulty']})")
            
            # Run RAG
            rag_res = self.evaluate_query(item, use_rag=True)
            results["rag"].append(rag_res)
            
            # Run Non-RAG
            non_rag_res = self.evaluate_query(item, use_rag=False)
            results["non_rag"].append(non_rag_res)
            
        return results

    def summarize_results(self, results):
        summary = []
        for mode in ["rag", "non_rag"]:
            df = pd.DataFrame(results[mode])
            accuracy = df['is_correct'].mean() * 100
            avg_latency = df['latency'].mean()
            
            # Accuracy by difficulty
            diff_acc = df.groupby('difficulty')['is_correct'].mean() * 100
            
            mode_summary = {
                "mode": mode,
                "overall_accuracy": accuracy,
                "avg_latency": avg_latency,
                "easy_acc": diff_acc.get('Easy', 0),
                "medium_acc": diff_acc.get('Medium', 0),
                "hard_acc": diff_acc.get('Hard', 0)
            }
            summary.append(mode_summary)
            
        return pd.DataFrame(summary), df # Return last df for failure analysis

if __name__ == "__main__":
    evaluator = PipelineEvaluator()
    results = evaluator.run_benchmark()
    summary_df, detailed_df = evaluator.summarize_results(results)
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
    
    # Save detailed results
    with open('evaluation_results_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Failure Analysis
    print("\n" + "="*50)
    print("FAILURE ANALYSIS (RAG MODE)")
    print("="*50)
    failures = detailed_df[detailed_df['is_correct'] == False]
    for _, f in failures.iterrows():
        print(f"ID: {f['query_id']} | Diff: {f['difficulty']}")
        print(f"Q: {f['question']}")
        print(f"Error: {f['status']}")
        print(f"Gen SQL: {f['generated_sql']}")
        print("-" * 30)
