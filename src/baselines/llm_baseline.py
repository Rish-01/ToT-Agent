from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# from huggingface_hub import InferenceClient
from tqdm import tqdm
import re
import os
from dotenv import load_dotenv



load_dotenv()  # loads variables from .env
# MY_API_TOKEN = os.environ.get("HF_TOKEN")  # read from environment variable

from google import genai


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def safe_text(response):
    if hasattr(response, "text") and response.text:
        return response.text
    try:
        return response.candidates[0].content.parts[0].text
    except:
        return ""


def run_math_llm(question):
    prompt = (
        "Answer the following grade-school math problem.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt
    )
    return safe_text(response).strip()


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
        
        if total >= 10:
            break

    accuracy = correct / total
    print("GSM8k baseline accuracy:", accuracy)

print("GSM8K: --------------------------------------------------------")
llm_gsm8k()

def extract_number_from_text(text):
    # Look for boxed answers
    m = re.search(r"\\boxed\{([0-9.+\-*/()]+)\}", text)
    if m:
        return m.group(1)

    # Look for phrases like "Final Answer: 10" or "Answer: 10"
    m = re.search(r"(Final Answer|Answer)\s*[:\- ]+\s*([0-9.+\-*/()]+)", text, re.IGNORECASE)
    if m:
        return m.group(2)

    # Look for standalone last number in the text
    m = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return m[-1]  # return the last number mentioned

    return None

    
def llm_MATH():

    # Load dataset
    dataset = load_dataset("EleutherAI/hendrycks_math", "algebra", streaming=True)
    test_stream = dataset["test"]

    # Evaluate: 
    correct = 0
    total = 0
    # subset = test.select(range(10))  # selects first 10 rows for testing
    for item in tqdm(test_stream):
        q = item["problem"]
        gold = item["solution"].strip()
        pred = run_math_llm(q)

        gold_num = extract_number_from_text(gold)
        pred_num = extract_number_from_text(pred)

        print("Gold: ", gold_num)
        print("Pred: ", pred_num)
        if gold_num is not None and pred_num is not None and gold_num == pred_num:
            correct += 1
            # print("yupppppppppppppppppppp")
            
        total += 1
        
        if total >= 10:
            break

    accuracy = correct / total
    print("MATH baseline accuracy:", accuracy)

print("MATH: ----------------------------------------------------------")
llm_MATH()


def run_commonsense_llm(question, options):
    prompt = (
        "Answer the following commonsense reasoning question.\n\n"
        f"Question: {question}\n"
        f"Options: {', '.join(options)}\n"
        "Answer with the letter of the correct option only:"
    )
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt
    )
    return safe_text(response).strip()


def extract_option_letter(text):
    """
    Extract the first occurrence of A/B/C/D as a standalone letter.
    Handles formats like:
    'A', 'A.', '(A)', 'Answer: A', 'The answer is C', etc.
    """
    m = re.search(r"\b([A-D])\b", text.upper())
    if m:
        return m.group(1)
    return None



def llm_ARC():
    # Use ARC-Challenge for evaluation (hard benchmark)
    dataset = load_dataset("ai2_arc", "ARC-Easy", streaming=True)
    test_stream = dataset["test"]

    correct = 0
    total = 0

    for item in tqdm(test_stream):
        q = item["question"]
        option_labels = item["choices"]["label"]
        option_texts = item["choices"]["text"]
        gold = item["answerKey"].strip().upper()

        # Create combined options like "A. Text"
        options = [f"{lbl}. {txt}" for lbl, txt in zip(option_labels, option_texts)]
        # print("RAW model output:", run_commonsense_llm(q, options))
        raw_pred = run_commonsense_llm(q, options)
        pred = extract_option_letter(raw_pred)

        print("Gold:", gold)
        print("Pred:", pred)

        if pred == gold:
            correct += 1
            print("Correct!")

        total += 1

        if total >= 10:
            break

    accuracy = correct / total
    print("ARC baseline accuracyi9:", accuracy)

print("ARC: --------------------------------------------------------------")
llm_ARC()

def run_knowledge_llm1(question):
    prompt = (
        "Answer the following question concisely based only on factual knowledge.\n\n"
        f"Question: {question}\nAnswer:"
    )
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt
    )
    return safe_text(response).strip()


def llm_HotPotQA():
    # Load dataset (validation split, streaming)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    
    correct = 0
    total = 0

    for item in tqdm(dataset):

        q = item["question"]
        gold = item["answer"].strip().lower()

        pred = run_knowledge_llm1(q).strip().lower()

        print("Gold:", gold)
        print("Pred:", pred)

        # SIMPLE SCORING: exact or substring match
        # (Exact match is too strict for free-text LLM answers)
        if gold in pred or pred in gold:
            correct += 1
            print("Correct!")

        total += 1

        # keep this for fast testing if needed
        if total >= 10:
            break

    accuracy = correct / total
    print("HotPotQA baseline accuracy:", accuracy)
    
print("hotpotqa: ----------------------------------------------------------------")
llm_HotPotQA()

def run_knowledge_llm2(question, options):
    prompt = (
        "Answer the following MMLU multiple-choice question.\n\n"
        f"Question: {question}\n"
        f"Options: {', '.join(options)}\n"
        "Answer with the letter of the correct option only:"
    )
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt
    )
    return safe_text(response).strip()


def llm_MMLU():
    dataset = load_dataset("cais/mmlu", "all", split="validation", streaming=True)
    test_stream = dataset

    correct = 0
    total = 0

    for item in tqdm(test_stream):
        q = item["question"]
        option_texts = item["choices"]
        gold_index = int(item["answer"])     # 0, 1, 2, 3

        # Convert 0/1/2/3 → A/B/C/D
        gold_letter = chr(ord("A") + gold_index)

        # Create strings like "A. option1"
        labels = ["A", "B", "C", "D"]
        options = [f"{lbl}. {txt}" for lbl, txt in zip(labels, option_texts)]

        raw_pred = run_knowledge_llm2(q, options)
        pred = extract_option_letter(raw_pred)


        print("Gold:", gold_letter)
        print("Pred:", pred)

        if pred == gold_letter:
            correct += 1
            print("Correct!")

        total += 1

        if total >= 10:
            break

    accuracy = correct / total
    print("MMLU baseline accuracy:", accuracy)

print("MMLU: ------------------------------------------------------------------")
llm_MMLU()