#!/usr/bin/env python3

import os
import re
from google.genai import types

import time
import json
import math
import logging
import datetime
import threading
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset
from google import genai

if "GEMMA_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GEMMA_API_KEY"] = ""


def normalize_answer(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\\boxed\{(.+?)\}", r"\1", s)
    s = re.sub(r"[^a-z0-9\s\-\.\,]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def token_overlap(a: str, b: str) -> float:
    ta = set(normalize_answer(a).split())
    tb = set(normalize_answer(b).split())
    return len(ta & tb) / max(1, len(ta | tb))

def extract_boxed_content(text: str) -> str:
    if not text: return ""
    idxs = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not idxs: return ""
    start = idxs[-1] + len("\\boxed{")
    cnt = 1
    end = start
    while end < len(text) and cnt > 0:
        if text[end] == '{': cnt += 1
        elif text[end] == '}': cnt -= 1
        end += 1
    return text[start:end-1]

def extract_answer_from_text(text: str) -> str:
    if not text: return ""
    
    boxed = extract_boxed_content(text)
    if boxed: return boxed

    finals = re.findall(r"Final Answer\s*[:\-]\s*(.+)", text, flags=re.I)
    if finals: return finals[-1].splitlines()[0].strip()
    
    answers = re.findall(r"Answer\s*[:\-]\s*(.+)", text, flags=re.I)
    if answers: return answers[-1].splitlines()[0].strip()

    last_block = text[-100:] 
    m = re.findall(r"\b([A-E])\b", last_block.upper())
    if m: return m[-1]

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()

def extract_hotpot_answer(text: str) -> str:
    if not text: return ""
    matches = re.findall(r"(?:final answer|answer is|ans:)\s*[:\-]?\s*(.*)", text, flags=re.I)
    if matches: return matches[-1].split("\n")[0].strip()
    
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()

def hotpot_match(pred: str, gold: str) -> bool:
    np = normalize_answer(pred)
    ng = normalize_answer(gold)
    if np == ng or np in ng or ng in np: return True
    if token_overlap(np, ng) > 0.7: return True
    return False


def eval_gsm8k(item, pred_text: str) -> int:
    gold = item.get("answer", "")
    gold_num = re.findall(r"-?\d+\.?\d*", gold.replace(",", ""))
    gold_num = gold_num[-1] if gold_num else ""
    
    pred = extract_answer_from_text(pred_text)
    pred_num = re.findall(r"-?\d+\.?\d*", pred.replace(",", ""))
    pred_num = pred_num[-1] if pred_num else ""
    
    try:
        return int(float(pred_num) == float(gold_num))
    except:
        return int(normalize_answer(pred) == normalize_answer(gold))

def eval_math(item, pred_text: str) -> int:
    gold_full = item.get("solution") or item.get("answer") or ""
    gold_ans = extract_boxed_content(gold_full)
    if not gold_ans: gold_ans = gold_full.split("\n")[-1]
    
    pred_ans = extract_answer_from_text(pred_text)
    
    if normalize_answer(pred_ans) == normalize_answer(gold_ans): return 1
    try:
        g_val = float(re.findall(r"-?\d+\.?\d*", gold_ans)[0])
        p_val = float(re.findall(r"-?\d+\.?\d*", pred_ans)[0])
        return int(abs(g_val - p_val) < 1e-6)
    except:
        return 0

def eval_multiple_choice(item, pred_text: str) -> int:
    gold = item.get("answerKey") or item.get("label") or item.get("answer", "")
    if str(gold).isdigit():
        gold = chr(65 + int(gold))
    gold = str(gold).strip().upper()
    
    pred = extract_answer_from_text(pred_text).upper()
    m = re.search(r"\b([A-E])\b", pred)
    pred_clean = m.group(1) if m else pred.strip()
    
    return int(pred_clean == gold)

def eval_hotpotqa(item, pred_text: str) -> int:
    gold = item.get("answer", "")
    pred = extract_hotpot_answer(pred_text)
    return int(hotpot_match(pred, gold))

def eval_logiqa(item, generation):
    if 'label' in item:
        idx = item['label']
        gold_char = chr(65 + idx)
    elif 'answer' in item: 
        gold_char = item['answer']
        idx = ord(gold_char) - 65
    else:
        print(f"Warning: neither 'label' nor 'answer' found in item keys: {item.keys()}")
        return False

    if idx < len(item['options']):
        raw_option = item['options'][idx]
        clean_text = re.sub(r'^[A-D][\.\)]\s*', '', raw_option)
    else:
        clean_text = "______"

    gen_lower = generation.lower()
    
    letter_match = f"answer: {gold_char.lower()}" in gen_lower
    text_match = clean_text.lower() in gen_lower
    
    return letter_match or text_match



def make_prompt_for_dataset(dataset_name: str, item: dict) -> str:
    name = dataset_name.lower()
    if "gsm8k" in name:
        return item["question"]
    if "math" in name:
        return item.get("problem", "") or item.get("question", "")
    if "logiqa" in name:
        q = item["question"]
        choices = item["options"]
        c = "\n".join([f"{chr(65+i)}. {t}" for i, t in enumerate(choices)])
        return f"{q}\n\nChoices:\n{c}"
    if "arc" in name or "ai2_arc" in name:
        q = item["question"]
        choices = item.get("choices", {})
        if isinstance(choices, dict):
            opts = choices.get("text", [])
            lbls = choices.get("label", [])
            c = "\n".join([f"{l}. {t}" for l, t in zip(lbls, opts)])
        else:
            c = str(choices)
        return f"{q}\n\nChoices:\n{c}"
    if "hotpot" in name:
        ctx = item.get("context") or item.get("supporting_facts", [])
        q = item["question"]
        if isinstance(ctx, list):
            ctx_text = "\n".join([" ".join(p) if isinstance(p, list) else str(p) for p in ctx][:5])
        else:
            ctx_text = str(ctx)[:4000]
        return f"Context:\n{ctx_text}\n\nQuestion:\n{q}"
    if "mmlu" in name:
        q = item["question"]
        choices = item.get("choices") or []
        c = "\n".join([f"{chr(65+i)}. {t}" for i, t in enumerate(choices)])
        return f"{q}\n\nChoices:\n{c}"
    return json.dumps(item)

def choose_evaluator_for_dataset(dataset_name: str) -> Callable:
    name = dataset_name.lower()
    if "gsm8k" in name: return eval_gsm8k
    if "math" in name: return eval_math
    if "logiqa" in name: return eval_multiple_choice
    if "arc" in name: return eval_multiple_choice
    if "hotpot" in name: return eval_hotpotqa
    if "mmlu" in name: return eval_multiple_choice
    return lambda item, pred: 0

def setup_logger():
    log_filename = f"logs/metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs("logs", exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


    logging.getLogger("google").setLevel(logging.ERROR)
    logging.getLogger("google.generativeai").setLevel(logging.ERROR)
    logging.getLogger("google.genai").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    return log_filename


def compute_confidence_interval(n: int, correct: int, confidence: float = 0.95) -> float:
    if n == 0: return 0.0
    p = correct / n
    z = 1.96 # Approx for 95%
    margin = z * math.sqrt((p * (1 - p)) / n)
    return margin

# -------------------------
# 5. CoT Agent
# -------------------------
class CoTAgent:
    def __init__(self, model: str = "gemma-3-27b-it", api_key_env: str = "GEMMA_API_KEY"):
        api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(f"Set {api_key_env} in your environment.")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model

        self.base_template = (
            "Solve the following problem step-by-step. Show your reasoning, then state "
            "the final answer prefixed with 'Answer:'.\n\n"
            "Problem:\n{problem}\n\n"
            "Reasoning steps:\n"
        )

        self.mmlu_template = (
            "Solve the following multiple-choice problem step-by-step. "
            "Analyze the choices, show your reasoning, and then state the final answer "
            "as a single letter (A, B, C, D, or E) prefixed with 'Answer:'.\n\n"
            "Problem:\n{problem}\n\n"
            "Reasoning steps:\n"
        )

    def generate_cot(self, problem_prompt: str, dataset_name: str) -> str:
        d_name = dataset_name.lower()
        if any(x in d_name for x in ["mmlu", "arc", "logiqa"]):
            full_prompt = self.mmlu_template.format(problem=problem_prompt)
        else:
            full_prompt = self.base_template.format(problem=problem_prompt)

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1024,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )
            return resp.text if resp.text else "Error: Empty response"
        except Exception as e:
            print(f"⚠️ Model error: {e}")
            time.sleep(10)
            return f"Error: {e}"

    def run_on_dataset(self, dataset_name: str, seed: int, split: str = "validation", config: Optional[str] = None, max_samples: Optional[int] = None):
        print(f"\n--- Loading {dataset_name} (Seed: {seed}) ---")
        try:
          
            print(dataset_name)
            print(seed)
            print(split)
            print(config)

            if config is not None:
                ds = load_dataset(
                    path=dataset_name,
                    name=config,
                    split=split
                )
            else:
                ds = load_dataset(
                    path=dataset_name,
                    split=split
    )
            ds = ds.shuffle(seed=seed)
        except Exception as e:
            logging.error(f"Error loading dataset {dataset_name}: {e}")
            return {}

        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))
        
        evaluator = choose_evaluator_for_dataset(dataset_name)
        correct_count = 0
        total = len(ds)

        for i, item in enumerate(ds):
            if(i%6==0):
                time.sleep(30)
            prompt = make_prompt_for_dataset(dataset_name, item)
            trace = self.generate_cot(prompt, dataset_name)
            
            correct = evaluator(item, trace)
            if correct:
                correct_count += 1
                
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{total}] Running Acc: {correct_count/(i+1):.2%}")

        accuracy = correct_count / total if total > 0 else 0
        ci_margin = compute_confidence_interval(total, correct_count)
        
        # Log Final Metrics
        log_msg = (
            f"DATASET: {dataset_name} | SEED: {seed} | SAMPLES: {total} | "
            f"ACCURACY: {accuracy:.4f} | 95% CI: [{max(0, accuracy - ci_margin):.4f}, {min(1, accuracy + ci_margin):.4f}] (+/- {ci_margin:.4f})"
        )
        logging.info(log_msg)
        
        return {
            "dataset": dataset_name,
            "seed": seed,
            "accuracy": accuracy,
            "ci_margin": ci_margin
        }


if __name__ == "__main__":
    setup_logger()
    
    datasets = [
        # ("openai/gsm8k", "main"),
        # ("hotpotqa/hotpot_qa", "distractor"),
        # ("ai2_arc", "ARC-Easy"),
        # ("lucasmccabe/logiqa", None),
        # ("EleutherAI/hendrycks_math", "algebra"),
        # ("cais/mmlu", "all"),
        ("lucasmccabe/logiqa","")
    ]
    
    seeds = [42, 43, 44] 
    agent = CoTAgent(model="gemma-3-27b-it")

    for ds_name, cfg in datasets:
        for seed in seeds:
            agent.run_on_dataset(ds_name, seed=seed, split="test", config=cfg, max_samples=500)
