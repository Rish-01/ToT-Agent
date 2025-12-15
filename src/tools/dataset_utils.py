import re
from datasets import load_dataset

from typing import Optional

def extract_answer_from_text(text: str) -> str:
    """Robust heuristic to find the answer in model output."""
    if not text: return ""
    
    # 1. Prefer Boxed (Strongest signal for Math)
    boxed = extract_boxed_content(text)
    if boxed: return boxed

    # 2. Explicit 'Final Answer' or 'Answer' tags
    finals = re.findall(r"Final Answer\s*[:\-]\s*(.+)", text, flags=re.I)
    if finals: return finals[-1].splitlines()[0].strip()
    
    answers = re.findall(r"Answer\s*[:\-]\s*(.+)", text, flags=re.I)
    if answers: return answers[-1].splitlines()[0].strip()

    # 3. Multiple Choice Fallback (A-E at end of text)
    last_block = text[-100:] 
    m = re.findall(r"\b([A-E])\b", last_block.upper())
    if m: return m[-1]

    # 4. Fallback: Last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def normalize_answer(s: Optional[str]) -> str:
    """Standardizes answer string for comparison."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\\boxed\{(.+?)\}", r"\1", s)
    s = re.sub(r"[^a-z0-9\s\-\.\,]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_boxed_content(text: str) -> str:
    """Extracts content inside \boxed{...}."""
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

class DatasetUtils:
    @staticmethod
    def load_and_prep_dataset(dataset_name, config=None, split="validation", max_samples=None, seed=42):
        """
        Loads the dataset and optionally selects a subset for testing.
        """
        print(f"Loading {dataset_name} (config={config}, split={split})...")
        try:
            # Handle special case for MMLU which often requires specific configs
            if "mmlu" in dataset_name.lower() and config == "all":
                 # Usually MMLU requires a specific subject config, 'all' might need special handling 
                 # or iterating over subsets, but here we pass it through if the user set it.
                 pass

            ds = load_dataset(dataset_name, config, split=split)
            ds = ds.shuffle(seed=seed)
            
            if max_samples:
                # Select the first N samples
                n = min(len(ds), max_samples)
                ds = ds.select(range(n))
                print(f"Selected {n} samples.")
            
            return ds
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    @staticmethod
    def make_prompt(dataset_name, item):
        """
        Constructs a prompt based on the specific dataset schema.
        """
        name = dataset_name.lower()

        # --- GSM8K ---
        if "gsm8k" in name:
            return f"Question: {item['question']}\nSolve this step by step."
        
        # --- MATH / Hendrycks Math ---
        elif "math" in name or "hendrycks" in name:
            return f"Problem: {item['problem']}\nSolve this step by step."
        
        # --- HotpotQA ---
        elif "hotpot" in name:
            # HotpotQA context is typically {'sentences': [['sent1', 'sent2'], ...]}
            context_data = item.get('context', {})
            
            if isinstance(context_data, dict):
                sentences_lists = context_data.get('sentences', [])
            else:
                sentences_lists = []
            
            # Flatten the list of lists
            flat_sentences = []
            for group in sentences_lists:
                if isinstance(group, list):
                    flat_sentences.extend(group)
                else:
                    flat_sentences.append(str(group))
            
            # Truncate to avoid context window issues (simple heuristic)
            context_text = " ".join(flat_sentences)[:3000]
            return f"Context: {context_text}\n\nQuestion: {item['question']}\nAnswer this question based on the context."
        
        # --- ARC (AI2_ARC) ---
        elif "arc" in name:
            # ARC choices format: {'text': ['A', 'B', ...], 'label': ['A', 'B', ...]}
            choices_text = item['choices']['text']
            # choices_label = item['choices']['label']
            
            formatted_choices = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices_text)])
            return f"Question: {item['question']}\n\nChoices:\n{formatted_choices}\n\nSelect the correct answer."
        
        # --- MMLU ---
        elif "mmlu" in name or "cais/mmlu" in name:
            # MMLU choices are usually a list of strings
            choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(item['choices'])])
            return f"Question: {item['question']}\n\nChoices:\n{choices}\n\nSelect the correct answer."
        
        # --- LogiQA ---
        elif "logiqa" in name:
            context = item.get('context', '')
            options = item.get('options', [])
            
            # Formatting options
            formatted_options = []
            for i, opt in enumerate(options):
                opt = str(opt).strip()
                if re.match(r'^[A-D][\.\)]', opt): 
                    formatted_options.append(opt)
                else:
                    formatted_options.append(f"{chr(65+i)}. {opt}")
                    
            return f"Context: {context}\n\nQuestion: {item['question']}\n\nOptions:\n" + "\n".join(formatted_options) + "\n\nProvide your reasoning and select the correct option."
        
        else:
            return str(item.get('question', item.get('problem', str(item))))


    @staticmethod
    def get_evaluator(dataset_name):
        """
        Returns a function(item, generation_text) -> bool
        """
        name = dataset_name.lower()

        if "gsm8k" in name:
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
            return eval_gsm8k
        
        # --- MATH Evaluator ---
        elif "math" in name or "hendrycks" in name:
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
            return eval_math

        
        # --- HotpotQA Evaluator ---
        elif "hotpot" in name:
            def eval_hotpot(item, generation):
                gold = item['answer'].lower().strip()
                return gold in generation.lower()
            return eval_hotpot
        
        # --- ARC Evaluator ---
        elif "arc" in name:
            def eval_arc(item, generation):
                gold_key = item['answerKey'] # e.g., "A" or "1"
                choices = item['choices']['text']
                
                # Resolve the text of the correct answer
                if gold_key.isdigit():
                    gold_text = choices[int(gold_key) - 1]
                elif len(gold_key) == 1 and 'A' <= gold_key <= 'Z':
                    idx = ord(gold_key) - ord('A')
                    gold_text = choices[idx]
                else:
                    gold_text = gold_key # Fallback
                
                return gold_text.lower() in generation.lower()
            return eval_arc
        
        # --- MMLU Evaluator ---
        elif "mmlu" in name or "cais/mmlu" in name:
            def eval_mmlu(item, generation):
                # item['answer'] is an integer index (0, 1, 2, 3)
                idx = item['answer']
                gold_text = item['choices'][idx]
                gold_char = chr(65 + idx)
                
                # Check for either the full text or "Answer: X" pattern
                # Being lenient here: if the text or the letter appears in a definitive context
                gen_lower = generation.lower()
                return (gold_text.lower() in gen_lower) or (f"answer: {gold_char.lower()}" in gen_lower)
            return eval_mmlu
        
        # --- LogiQA Evaluator ---
        elif "logiqa" in name:
            def eval_logiqa(item, generation):
                # Handle different column names dynamically
                if 'label' in item: # Old dataset
                    idx = item['label']
                    gold_char = chr(65 + idx)
                elif 'answer' in item: # New dataset (fireworks-ai)
                    gold_char = item['answer']
                    idx = ord(gold_char) - 65
                else:
                    print(f"Warning: neither 'label' nor 'answer' found in item keys: {item.keys()}")
                    return False

                # Handle bounds checking
                if idx < len(item['options']):
                    raw_option = item['options'][idx]
                    clean_text = re.sub(r'^[A-D][\.\)]\s*', '', raw_option)
                else:
                    clean_text = "______"

                gen_lower = generation.lower()
                
                # Check for "Answer: A" or the full text
                letter_match = f"answer: {gold_char.lower()}" in gen_lower
                text_match = clean_text.lower() in gen_lower
                
                return letter_match or text_match
            return eval_logiqa
        
        # --- Default Evaluator ---
        else:
            def eval_default(item, generation):
                # Try to find a generic answer field
                gold = str(item.get('answer', item.get('solution', ''))).strip()
                return gold in generation if gold else False
            return eval_default