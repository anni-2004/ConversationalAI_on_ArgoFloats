import json
import pandas as pd
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_executor import AgenticSQLQueryExecutor
from database_manager import DatabaseManager

class AgentEvaluator:
    def __init__(self):
        self.executor = AgenticSQLQueryExecutor()
        self.db_manager = DatabaseManager()
        self.dataset = self.load_dataset()
    def load_dataset(self):
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation_dataset.json')
        with open(dataset_path, 'r') as f:
            return json.load(f)

    def execute_and_compare(self, generated_sql, ground_truth_sql):
        try:
            gen_res = self.db_manager.execute_query(generated_sql)
            gt_res = self.db_manager.execute_query(ground_truth_sql)
            
            if not gen_res["success"]:
                return False
            
            df_gen = pd.DataFrame(gen_res["data"])
            df_gt = pd.DataFrame(gt_res["data"])
            
            if df_gen.equals(df_gt):
                return True
            
            # Simple content match
            if len(df_gen) == len(df_gt) and len(df_gen.columns) == len(df_gt.columns):
                try:
                    df_gen_sorted = df_gen.sort_values(by=list(df_gen.columns)).reset_index(drop=True)
                    df_gt_sorted = df_gt.sort_values(by=list(df_gt.columns)).reset_index(drop=True)
                    if (df_gen_sorted.values == df_gt_sorted.values).all():
                        return True
                except:
                    pass
            return False
        except:
            return False

    def run_evaluation(self):
        results = []
        print(f"Starting Agentic Evaluation on {len(self.dataset)} queries...")
        
        for i, item in enumerate(self.dataset):
            print(f"[{i+1}/{len(self.dataset)}] Agentic Query: {item['id']} ({item['difficulty']})")
            
            start_time = time.time()
            response = self.executor.query_with_correction(item['question'])
            latency = time.time() - start_time
            
            # Re-check accuracy for final result
            is_correct = False
            if response.get("status") in ["Success (First Pass)", "Success (After Correction)", "Failed"] and "generated_query" in response:
                is_correct = self.execute_and_compare(response["generated_query"], item['ground_truth_sql'])

            res_entry = {
                "id": item['id'],
                "difficulty": item['difficulty'],
                "question": item['question'],
                "status": response.get("status"),
                "is_correct": is_correct,
                "latency": latency
            }
            results.append(res_entry)
            
        return results

    def print_summary(self, results):
        df = pd.DataFrame(results)
        
        # Overall Stats
        total = len(df)
        first_pass_success = len(df[df['status'] == 'Success (First Pass)'])
        recovered = len(df[df['status'] == 'Success (After Correction)'])
        total_success = len(df[df['is_correct'] == True])
        
        print("\n" + "="*50)
        print("AGENTIC EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Queries: {total}")
        print(f"First-Pass Accuracy: {(first_pass_success/total)*100:.1f}%")
        print(f"Recovery Rate (Fixed in Pass 2): {(recovered/total)*100:.1f}%")
        print(f"Final Execution Accuracy: {(total_success/total)*100:.1f}%")
        print("-" * 30)
        
        # Accuracy by difficulty
        diff_acc = df.groupby('difficulty')['is_correct'].mean() * 100
        print("Accuracy by Difficulty:")
        print(diff_acc.to_string())
        print("="*50)

if __name__ == "__main__":
    evaluator = AgentEvaluator()
    results = evaluator.run_evaluation()
    evaluator.print_summary(results)
    
    # Save results
    with open('agentic_rag/agentic_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
