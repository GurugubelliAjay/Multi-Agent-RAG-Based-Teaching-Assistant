import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
import time
import re
import warnings

# Ignore Ragas deprecation warnings to keep the console clean
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq

# Import your existing tools
from agent_tools import get_embeddings, agentic_rag_response

def run_crag_evaluation():
    username = "ajay"
    subject = "Data Structures and Algorithms"
    
    test_cases = [
        {"question": "What are the essential properties of an algorithm?", "ground_truth": "An algorithm must possess finiteness, definiteness, effectiveness, generality and input/output."},
        {"question": "What is time complexity?", "ground_truth": "The time complexity of an algorithm is a function of the running time of the algorithm."},
        {"question": "What is the purpose of asymptotic notation?", "ground_truth": "Asymptotic notation is a shorthand way to represent the time complexity of an algorithm."},
        {"question": "Define Big-O notation with an example.", "ground_truth": "Big Oh notation is a method of representing the upper bound of an algorithm's running time. Example: O(n) linear."},
        {"question": "Differentiate best case, worst case, and average case complexity.", "ground_truth": "Best case complexity is the minimum time to run, worst case complexity is the maximum time to run, and average case complexity is the average time for given inputs."},
        {"question": "What are the two main components of space complexity?", "ground_truth": "Space complexity consists of a constant part and a variable part depending on instance characteristics."},
        {"question": "What is a Binary Search Tree (BST)?", "ground_truth": "A Binary Search Tree is a binary tree in which elements in the left subtree are less than the root and elements in the right subtree are greater than the root."},
        {"question": "How does binary search find a target value?", "ground_truth": "Binary search works by comparing the middle element and dividing the list into sub-arrays until the element is found."},
        {"question": "Why must data be sorted for binary search?", "ground_truth": "Before applying binary search, the list of items should be sorted in ascending or descending order."},
        {"question": "Compare linear search and binary search in terms of efficiency.", "ground_truth": "Linear search has time complexity O(n) while binary search has time complexity O(log n)."},
        {"question": "What is the main advantage of a doubly linked list?", "ground_truth": "The main advantage of a doubly linked list is that it permits traversing or searching of the list in both directions."},
        {"question": "How does selection sort organize elements?", "ground_truth": "Selection sort selects the smallest element from the list and places it in the correct position."},
        {"question": "How does insertion sort build a sorted list?", "ground_truth": "Insertion sort removes one element and inserts it into its correct position in the sorted list."},
        {"question": "What algorithm design technique does merge sort use?", "ground_truth": "Merge sort uses divide and conquer technique."},
        {"question": "What is the role of the pivot in quick sort?", "ground_truth": "The pivot is used to partition the array into elements less than and greater than it."},
        {"question": "What is the purpose of a data structure?", "ground_truth": "A data structure is a specialized format for organizing and storing data so that it can be accessed and worked with efficiently."},
        {"question": "What is an Abstract Data Type (ADT)?", "ground_truth": "An abstract data type is a data type defined by its behavior from the point of view of a user, without specifying implementation details."},
        {"question": "What is the primary advantage of a linked list over an array?", "ground_truth": "A linked list is a flexible dynamic data structure in which elements can be added or deleted easily."},
        {"question": "What causes a stack overflow?", "ground_truth": "Stack overflow occurs when we try to push more elements onto the stack than it can hold."},
        {"question": "How does a queue differ from a stack?", "ground_truth": "A queue follows first in first out order whereas a stack follows last in first out order."}
    ]
    
    data_crag = {
        "user_input": [], 
        "response": [], 
        "retrieved_contexts": [], 
        "reference": []
    }
    
    print(f"Running {len(test_cases)} test cases for the CRAG Baseline...")
    print(f"\n--- Gathering Responses for CRAG Baseline ---")
    for tc in test_cases:
        query = tc["question"]
        print(f"Processing (CRAG): {query}")
        
        response = None
        contexts = []
        for attempt in range(5):
            try:
                # Notice max_critic_loops=0 to simulate CRAG
                res, ctxs = agentic_rag_response(username, subject, query, return_raw=True, max_critic_loops=0)
                response = res.answer
                contexts = ctxs
                break
            except Exception as e:
                err_str = str(e)
                delay = 10.0
                if "429" in err_str:
                    match_groq = re.search(r'(?:([0-9]+)h)?(?:([0-9]+)m)?([0-9.]+)s', err_str)
                    if match_groq:
                        h = float(match_groq.group(1)) if match_groq.group(1) else 0.0
                        m = float(match_groq.group(2)) if match_groq.group(2) else 0.0
                        s = float(match_groq.group(3)) if match_groq.group(3) else 0.0
                        delay = (h * 3600) + (m * 60) + s + 5.0
                
                print(f"  [GENERATION ERROR] Attempt {attempt+1}/5 failed: {e.__class__.__name__}. Retrying after {delay:.1f}s...")
                time.sleep(delay)
                    
        if not response:
            print(f"  [ERROR] Exhausted retries for query: {query}. Skipping.")
            continue
        
        data_crag["user_input"].append(query)
        data_crag["response"].append(response)
        data_crag["retrieved_contexts"].append(contexts)
        data_crag["reference"].append(tc["ground_truth"])
        
    dataset_crag = Dataset.from_dict(data_crag)
    
    # Use Llama 4 Scout 17B for its massive 30k TPM limit to survive Ragas payloads
    eval_llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
        temperature=0,
        max_retries=7,
        timeout=240.0,
        max_tokens=4096 
    )
    ragas_llm = LangchainLLMWrapper(eval_llm)
    eval_embeddings = get_embeddings()
    ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)
    
    config = RunConfig(max_workers=1, timeout=240, max_retries=10)
    
    flat_metrics = [
        Faithfulness(llm=ragas_llm), 
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb, strictness=1), 
        ContextPrecision(llm=ragas_llm), 
        ContextRecall(llm=ragas_llm)     
    ]
    
    print(f"\nStarting Ragas Bulk Evaluation for CRAG Baseline... (This may take a minute)")
    all_results = []
    for i in range(len(dataset_crag)):
        single_row = dataset_crag.select([i])
        print(f"\nEvaluating CRAG Baseline - Question {i+1}/{len(dataset_crag)}...")
        
        for attempt in range(5):
            try:
                res = evaluate(
                    dataset=single_row,
                    metrics=flat_metrics,
                    run_config=config,
                    raise_exceptions=True
                )
                all_results.append(res.to_pandas())
                break
            except Exception as e:
                err_str = str(e)
                if "413" in err_str or "Request too large" in err_str:
                    print(f"  [FATAL ERROR] Payload too large for Groq's 6000 TPM limit. Marking metrics as NaN.")
                    dummy_df = single_row.to_pandas()
                    for m in flat_metrics:
                        dummy_df[getattr(m, 'name', m.__class__.__name__)] = float('nan')
                    all_results.append(dummy_df)
                    break
                elif "quota_metric" in err_str or "429" in err_str or "ResourceExhausted" in err_str:
                    delay = 60.0
                    match_google = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}', err_str)
                    match_groq = re.search(r'(?:([0-9]+)h)?(?:([0-9]+)m)?([0-9.]+)s', err_str)
                    if match_google:
                        delay = float(match_google.group(1)) + 5.0
                    elif match_groq:
                        h = float(match_groq.group(1)) if match_groq.group(1) else 0.0
                        m = float(match_groq.group(2)) if match_groq.group(2) else 0.0
                        s = float(match_groq.group(3)) if match_groq.group(3) else 0.0
                        delay = (h * 3600) + (m * 60) + s + 5.0
                    print(f"  [RATE LIMIT] API limit exceeded. Sleeping for {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"  [EVAL ERROR] Attempt {attempt+1}/5 failed: {e.__class__.__name__} - {e}")
                    if attempt >= 4:
                        print(f"  [FATAL ERROR] Skipping Question {i+1} due to persistent errors.")
                        dummy_df = single_row.to_pandas()
                        for m in flat_metrics:
                            dummy_df[getattr(m, 'name', m.__class__.__name__)] = float('nan')
                        all_results.append(dummy_df)
                        break
                    else:
                        print("  Retrying in 10 seconds...")
                        time.sleep(10)
                
        time.sleep(15)
        time.sleep(65)
        
        if all_results:
            df_crag = pd.concat(all_results, ignore_index=True)
            df_crag.to_csv("ragas_crag_results.csv", index=False)
            print("--> Progressively saved CRAG results to 'ragas_crag_results.csv'")
    
    df_crag = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    print("\n--- EVALUATION RESULTS ---")
    if not df_crag.empty:
        print("Detailed results saved to 'ragas_crag_results.csv'.")
        
        metrics_to_compare = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        
        print("\n=== CRAG BASELINE SUMMARY ===")
        print(f"Total Questions Evaluated: {len(df_crag)}")
        for m in metrics_to_compare:
            if m in df_crag.columns:
                val = df_crag[m].mean()
                print(f"{m.replace('_', ' ').title():<25} | {val:<12.4f}")
        print("=========================\n")
    else:
        print("Evaluation failed. No results were generated.")

if __name__ == "__main__":
    run_crag_evaluation()