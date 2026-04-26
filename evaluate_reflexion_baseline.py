import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
import time
import re
import warnings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
load_dotenv()

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from agent_tools import get_embeddings, agentic_rag_response, safe_invoke

def run_reflexion_evaluation():
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
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    data_reflexion = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": [], "latency": []}
    
    print(f"\n--- Gathering Responses for Reflexion Baseline ---")
    for idx, tc in enumerate(test_cases):
        query = tc["question"]
        print(f"\nProcessing {idx+1}/20: {query}")
        
        
        final_answer = None
        contexts = []
        latency = 0.0
        
        for attempt in range(5):
            try:
                start_time = time.time()
                
                # Step 1 & 2: Heavy Retrieve & Draft (matches Proposed RAG's first half)
                res, ctxs = agentic_rag_response(username, subject, query, return_raw=True, max_critic_loops=0)
                draft = res.answer
                contexts = ctxs
                context_text = "\n\n".join(contexts)
                
                # TRUE REFLEXION LOOP
                current_answer = draft
                max_turns = 3
                turn = 0
                
                while turn < max_turns:
                    # Step 3: Critique
                    critique_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a harsh grader. Check if the draft answer contains any facts NOT present in the context. If it does, point them out. If perfectly grounded, output 'PASS'.\nContext: {context}"),
                        ("human", "Question: {question}\nDraft: {draft}")
                    ])
                    critique = safe_invoke(critique_prompt | llm, {"context": context_text, "question": query, "draft": current_answer})
                    
                    if "PASS" in critique:
                        print(f"    [Turn {turn+1}] Critic passed.")
                        break
                        
                    print(f"    [Turn {turn+1}] Critic found issues. Refining...")
                    
                    # Step 4: Refine
                    refine_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Rewrite the draft answer to fix the issues raised in the critique. Ensure NO outside knowledge is used.\nContext: {context}"),
                        ("human", "Draft: {draft}\nCritique: {critique}")
                    ])
                    current_answer = safe_invoke(refine_prompt | llm, {"context": context_text, "draft": current_answer, "critique": critique})
                    turn += 1
                    
                final_answer = current_answer
                
                # Fallback if the model explicitly says it can't answer
                if "cannot find the answer" in final_answer.lower() or "not present in the context" in final_answer.lower() or "insufficient context" in final_answer.lower():
                    final_answer = "I don't know."
                    
                latency = time.time() - start_time
                print(f"  Latency: {latency:.2f}s | True Reflexion executed ({turn} refinements).")
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
                
        if not final_answer:
            print(f"  [ERROR] Exhausted retries for query: {query}. Skipping.")
            continue
        
        data_reflexion["user_input"].append(query)
        data_reflexion["response"].append(final_answer)
        data_reflexion["retrieved_contexts"].append(contexts)
        data_reflexion["reference"].append(tc["ground_truth"])
        data_reflexion["latency"].append(latency)
        
    avg_latency = sum(data_reflexion["latency"]) / len(data_reflexion["latency"])
    print(f"\n---> Average Reflexion Inference Latency: {avg_latency:.2f} seconds")
    
    # --- START RAGAS EVALUATION ---
    dataset_reflexion = Dataset.from_dict(data_reflexion)
    
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
    
    print(f"\nStarting Ragas Evaluation on Reflexion answers...")
    all_results = []
    for i in range(len(dataset_reflexion)):
        single_row = dataset_reflexion.select([i])
        print(f"\nEvaluating Question {i+1}/{len(dataset_reflexion)}...")
        
        row_df = single_row.to_pandas()
        
        for metric in flat_metrics:
            print(f"  -> Evaluating {metric.__class__.__name__}...")
            for attempt in range(5):
                try:
                    res = evaluate(dataset=single_row, metrics=[metric], run_config=config, raise_exceptions=True)
                    res_df = res.to_pandas()
                    
                    # Merge the newly evaluated metric column into our row
                    for col in res_df.columns:
                        if col not in row_df.columns:
                            row_df[col] = res_df[col]
                    break
                except Exception as e:
                    err_str = str(e)
                    if "413" in err_str or "Request too large" in err_str:
                        print(f"    [FATAL ERROR] Payload too large for Groq. Marking metric as NaN.")
                        row_df[getattr(metric, 'name', metric.__class__.__name__)] = float('nan')
                        break
                    elif "429" in err_str or "quota" in err_str.lower() or "ResourceExhausted" in err_str:
                        print(f"    [RATE LIMIT] Sleeping for 65s...")
                        time.sleep(65)
                    else:
                        print(f"    [ERROR] Attempt {attempt+1}/5 failed: {e.__class__.__name__}")
                        if attempt >= 4:
                            print(f"    [FATAL ERROR] Skipping metric due to persistent errors. Marking as NaN.")
                            row_df[getattr(metric, 'name', metric.__class__.__name__)] = float('nan')
                            break
                        else:
                            time.sleep(10)
            time.sleep(5) # Short pause between metrics to pace the token bucket
                
        all_results.append(row_df)
        
        if all_results:
            df_reflexion = pd.concat(all_results, ignore_index=True)
            df_reflexion.to_csv("ragas_reflexion_results.csv", index=False)
            print(f"--> Saved progress to 'ragas_reflexion_results.csv'")
            
        time.sleep(10) # Pause before the next question
            
    df_reflexion = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    print("\n--- EVALUATION RESULTS ---")
    if not df_reflexion.empty:
        metrics_to_compare = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        print("\n=== REFLEXION BASELINE SUMMARY ===")
        print(f"Total Questions Evaluated: {len(df_reflexion)}")
        for m in metrics_to_compare:
            if m in df_reflexion.columns:
                val = df_reflexion[m].mean()
                print(f"{m.replace('_', ' ').title():<25} | {val:<12.4f}")
        print(f"{'Average Latency':<25} | {avg_latency:<12.2f} seconds")
        print("=========================\n")

if __name__ == "__main__":
    run_reflexion_evaluation()