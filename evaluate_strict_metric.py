import pandas as pd
import time
import ast
from agent_tools import check_hallucinations

def evaluate_ssf(csv_filename, name):
    print(f"\n--- Evaluating Strict Syllabus Faithfulness (SSF) for {name} ---")
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Could not read {csv_filename}: {e}")
        return
        
    # Ignore safe refusals from the evaluation to get the "Answered-Only" accuracy
    answered_df = df[df['response'] != "I don't know."]
    total = len(answered_df)
    passed = 0
    
    for idx, row in answered_df.iterrows():
        # The CSV stores the context as a string representation of a list. We need to parse it back.
        try:
            context_list = ast.literal_eval(row['retrieved_contexts'])
            context_text = "\n\n".join(context_list)
        except (ValueError, SyntaxError):
            context_text = str(row['retrieved_contexts'])
            
        answer = row['response']
        
        # Send it to YOUR strict Gate 2 Critic!
        grounded, reason = check_hallucinations(context_text, answer)
        
        if grounded:
            passed += 1
            print(f"  [Q{idx+1}] PASS")
        else:
            print(f"  [Q{idx+1}] FAIL (Hallucinated outside knowledge): {reason}")
        
        # Sleep briefly to avoid Groq rate limits
        time.sleep(2)
        
    score = (passed / total) if total > 0 else 0
    print(f"\n=> {name} Final SSF Score: {score:.4f} ({passed}/{total} grounded)")

if __name__ == "__main__":
    evaluate_ssf("ragas_crag_results.csv", "CRAG Baseline")
    evaluate_ssf("ragas_proposed_results.csv", "Proposed RAG")