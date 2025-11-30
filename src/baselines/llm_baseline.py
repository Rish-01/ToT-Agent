from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import InferenceClient
from tqdm import tqdm
import re
import os
from dotenv import load_dotenv



load_dotenv()  # loads variables from .env
MY_API_TOKEN = os.environ.get("HF_TOKEN")  # read from environment variable
print(MY_API_TOKEN)

client = InferenceClient(api_key=MY_API_TOKEN)


# downloads huggingface gemma3 model

model_name = "google/gemma-3-27b-it"



def run_math_llm(question):
    prompt = f"Answer the following grade-school math problem.\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.0,
    )

    return response.choices[0].message["content"]

def extract_answer_portion(pred_text):
    """
    Returns everything after '**Answer:**' in the prediction.
    """
    parts = pred_text.split("**Answer:**")
    if len(parts) < 2:
        return ""  # fallback if no **Answer:**
    return parts[1]

def normalize_number_str(num_str):
    """
    Remove commas, spaces, and other non-digit characters for comparison.
    """
    return re.sub(r"[^\d]", "", num_str)

def llm_gsm8k():

    # Load dataset
    dataset = load_dataset("gsm8k", "main", streaming=True)
    test_stream = dataset["test"]

    # Evaluate: 
    correct = 0
    total = 0
    # subset = test.select(range(10))  # selects first 10 rows for testing
    for item in tqdm(test_stream):
        q = item["question"]
        gold = item["answer"].strip()

        pred = run_math_llm(q)

        # extract gold number and normalize
        gold_number = re.search(r"####\s*([0-9,.\-]+)", gold).group(1)
        gold_number_norm = normalize_number_str(gold_number)

        # extract answer portion and normalize all numbers inside it
        answer = extract_answer_portion(pred)
        pred_norm = re.sub(r"[0-9,.\-]+", lambda m: normalize_number_str(m.group(0)), answer)


        # print("Gold: ", gold_number_norm)
        # print("Pred: ", pred_norm)
        if gold_number_norm in pred_norm:
            correct += 1
            # print("yupppppppppppppppppppp")
            
        total += 1

    accuracy = correct / total
    print("GSM8k baseline accuracy:", accuracy)


def run_symbolic_llm(question, options):
    prompt = f"Answer the following symbolic reasoning question:\n\nQuestion: {question}\nOptions: {', '.join(options)}\nAnswer with the letter of the correct option only:"
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.0,
    )
    
    return response.choices[0].message["content"].strip()


def llm_logiQA():
    
    dataset = load_dataset("logiqa", "main", streaming=True)
    test_stream = dataset["test"]
    correct = 0
    total = 0
    
    for item in tqdm(test_stream):
        q = item["question"]
        options = item["options"]
        gold = item["answer"].strip().upper()  # usually "A", "B", etc.
        
        pred = run_symbolic_llm(q, options).upper()
        
        print("Gold:", gold)
        print("Pred:", pred)
        
        if pred == gold:
            correct += 1
            print("Correct!")
        
        total += 1
        
        if total >= 10:
            break
    accuracy = correct/total
    print("Symbolic baseline accuracy: accuracy")

llm_logiQA()
