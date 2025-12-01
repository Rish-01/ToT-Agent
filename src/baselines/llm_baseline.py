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

import google.genai as genai


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

# print("GSM8K: --------------------------------------------------------")
# llm_gsm8k()



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
        gold_raw = item["answerKey"]
        gold = gold_raw.strip().upper()
        # Create combined options like "A. Text"
        options = [f"{lbl}. {txt}" for lbl, txt in zip(option_labels, option_texts)]
        # print("RAW model output:", run_commonsense_llm(q, options))
        raw_pred = run_commonsense_llm(q, options)
        pred = extract_option_letter(raw_pred)

        # print("Gold:", gold_raw)
        # print("Pred:", raw_pred)

        if pred == gold:
            correct += 1
            print("Correct!")

        total += 1

        if total >= 10:
            break

    accuracy = correct / total
    print("ARC baseline accuracy:", accuracy)

# print("ARC: --------------------------------------------------------------")
# llm_ARC()

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

def normalize(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)      # collapse spaces
    return text

def hotpotqa_match(gold, pred):
    gold_norm = normalize(gold)
    pred_norm = normalize(pred)
    print("Gold_norm: ", gold_norm)
    print("Pred norm: ", pred_norm)
    # exact match
    if gold_norm == pred_norm:
        return True

    # substring match
    if gold_norm in pred_norm:
        return True

    # token overlap: if ≥50% of gold tokens appear in prediction
    gold_tokens = set(gold_norm.split())
    pred_tokens = set(pred_norm.split())
    overlap = len(gold_tokens & pred_tokens) / len(gold_tokens)
    if overlap >= 0.5:
        return True

    return False


def llm_HotPotQA():
    # Load dataset (validation split, streaming)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    
    correct = 0
    total = 0

    for item in tqdm(dataset):

        q = item["question"]
        gold_raw = item["answer"]
        # gold = gold_raw.strip().lower()

        pred_raw = run_knowledge_llm1(q)
        # pred = pred_raw.strip().lower()

        print("Gold:", gold_raw)
        print("Pred:", pred_raw)

        # SIMPLE SCORING: exact or substring match
        # (Exact match is too strict for free-text LLM answers)
        if hotpotqa_match(gold_raw, pred_raw):
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


        # print("Gold:", gold_index)
        # print("Pred:", raw_pred)

        if pred == gold_letter:
            correct += 1
            print("Correct!")

        total += 1

        if total >= 10:
            break

    accuracy = correct / total
    print("MMLU baseline accuracy:", accuracy)

# print("MMLU: ------------------------------------------------------------------")
# llm_MMLU()