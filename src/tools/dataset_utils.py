import re
from datasets import load_dataset

def extract_answer_from_text(text: str) -> str:
    """
    Extracts the answer after 'Answer:' or returns the last sentence.
    """
    # 1. Look for "Answer: <content>"
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    
    # 2. Fallback: Look for "The answer is <content>"
    if "The answer is" in text:
        return text.split("The answer is")[-1].strip()
    
    # 3. Fallback: Return the last non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines[-1] if lines else ""

def normalize_answer(s: str) -> str:
    """
    Lowercases, removes trailing punctuation, and handles currency/formatting.
    """
    if not s: return ""
    s = s.lower().strip()
    # Remove final punctuation
    if s and s[-1] in ".,;:!?":
        s = s[:-1]
    # Remove commas from numbers (e.g. 1,000 -> 1000)
    s = s.replace(",", "")
    # Remove currency symbols
    s = s.replace("$", "")
    return s.strip()

def extract_boxed_content(text: str) -> str:
    """
    Extracts content inside \boxed{...}. Handles nested braces.
    """
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    
    # Start looking after \boxed{
    idx += 7 
    balance = 1
    content = []
    
    for char in text[idx:]:
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
        
        if balance == 0:
            break
        content.append(char)
        
    return "".join(content)

class DatasetUtils:
    @staticmethod
    def load_and_prep_dataset(dataset_name, config=None, split="validation", max_samples=None):
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
                # 1. Parse GOLD (Use #### delimiter)
                gold_raw = item.get("answer", "")
                if "####" in gold_raw:
                    gold_str = gold_raw.split("####")[-1].strip()
                else:
                    gold_str = gold_raw
                
                # 2. Parse PREDICTION
                pred_str = extract_answer_from_text(pred_text)

                # 3. Numeric Extraction
                # Remove commas to ensure 1,000 parses as 1000
                gold_nums = re.findall(r"-?\d+\.?\d*", gold_str.replace(",", ""))
                pred_nums = re.findall(r"-?\d+\.?\d*", pred_str.replace(",", ""))
                
                gold_val = float(gold_nums[-1]) if gold_nums else None
                pred_val = float(pred_nums[-1]) if pred_nums else None

                # 4. Compare
                # A: Float Equality
                if gold_val is not None and pred_val is not None:
                    if abs(gold_val - pred_val) < 1e-6:
                        return 1
                
                # B: String Normalization fallback (for non-numeric answers)
                return int(normalize_answer(pred_str) == normalize_answer(gold_str))
            
            return eval_gsm8k
        
        # --- MATH Evaluator ---
        elif "math" in name or "hendrycks" in name:
            def eval_math(item, pred_text: str) -> int:
                gold_full = item.get("solution") or item.get("answer") or ""
            
                # 1. Parse GOLD (Prioritize \boxed)
                gold_ans = extract_boxed_content(gold_full)
                if not gold_ans:
                    # Fallback for MATH: usually the answer is widely accepted as the last part
                    gold_ans = gold_full.split("\n")[-1]

                # 2. Parse PREDICTION
                pred_ans = extract_answer_from_text(pred_text)
                
                # 3. Compare Normalized Strings (Exact Match)
                if normalize_answer(pred_ans) == normalize_answer(gold_ans):
                    return 1
                    
                # 4. Compare Numeric Values (if both are numbers)
                try:
                    g_nums = re.findall(r"-?\d+\.?\d*", gold_ans.replace(",", ""))
                    p_nums = re.findall(r"-?\d+\.?\d*", pred_ans.replace(",", ""))
                    
                    if g_nums and p_nums:
                        # Compare the last numbers found in the boxed/extracted text
                        g_val = float(g_nums[-1])
                        p_val = float(p_nums[-1])
                        return int(abs(g_val - p_val) < 1e-6)
                except:
                    pass

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