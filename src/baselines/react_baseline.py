import os
import re
import time
import pandas as pd
from google import genai
from datasets import load_dataset

from src.tools.math_tools import EquationSolverTool, CalculatorTool
from src.tools.retriever import WikipediaTool
from src.tools.dataset_utils import DatasetUtils

class ReActAgent:
    def __init__(self, model="gemma-3-27b-it"):
        api_key = os.environ["GEMMA_API_KEY"]
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_steps = 5

        self.tools = {
            "equation_solver": EquationSolverTool(),
            "calculator": CalculatorTool(),
            "wiki": WikipediaTool()
        }

    def react_step(self, prompt):
        current_state = prompt
        
        for step in range(self.max_steps):
            prompt = get_react_prompt(current_state)
            
            while True:
                try:
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config={
                            "temperature": 0.5, 
                            "top_p": 0.9,
                        }
                    )
                    step_output = resp.text
                    break
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        print(f"Rate limit hit! Waiting 60 seconds...")
                        time.sleep(60)
                    else:
                        return current_state + f"\nERROR: {e}"

            # Process tool calls in the thought
            processed_output = self.resolve_tool_calls(step_output)
            current_state += f"\nStep {step+1}: {processed_output}"
            
            # Terminate if answer is found
            if "Answer:" in step_output:
                return current_state

        return current_state


    def resolve_tool_calls(self, text: str) -> str:
        """
        Scans text for TOOL_CALL:name:input, executes them, 
        and appends TOOL_OUTPUT:result.
        """
        # Regex to match: TOOL_CALL:toolname:input string (until end of line)
        pattern = r"TOOL_CALL:(\w+):([^\n]+)"
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return text

        modified_text = text
        
        for match in matches:
            tool_name = match.group(1).lower().strip()
            tool_input = match.group(2).strip()
            
            # Execute tool
            if tool_name in self.tools:
                print(f"[Tool Exec] {tool_name} with input: {tool_input}")
                try:
                    result = self.tools[tool_name].run(tool_input)
                except Exception as e:
                    result = f"Error executing tool: {e}"
            else:
                result = f"Error: Tool '{tool_name}' not found."
            
            # Append output
            modified_text += f"\nTOOL_OUTPUT:{result}"
            
        return modified_text

    def run_eval(self, dataset_name, split="test", config=None, max_samples=None):
        ds = DatasetUtils.load_and_prep_dataset(dataset_name, config, split, max_samples)
        evaluator = DatasetUtils.get_evaluator(dataset_name)
        
        results = []
        
        for i, item in enumerate(ds):
            prompt = DatasetUtils.make_prompt(dataset_name, item)
            # Ensure directory exists for visualization            
            try:
                print(f"\nProcessing Sample {i+1}...")
                final_state = self.react_step(prompt)
                prediction = final_state 
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                prediction = ""
            
            is_correct = evaluator(item, prediction)
            print(f"Correct: {is_correct}")
            
            results.append({
                "index": i,
                "correct": is_correct,
                "prediction": prediction
            })
        
        # Calculate final metrics
        accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
        
        # Create the summary dictionary your loop expects
        summary_res = {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "n": len(results),
            "results": results
        }
        
        # Return only the dictionary
        return summary_res


def get_react_prompt(state: str) -> str:
    base_prompt = f"""
        You are a ReAct (Reasoning + Acting) agent.
        Solve the user's problem by interleaving Thought, Action, and Observation steps.

        Your job is to expand reasoning states into multiple possible next thoughts. 
        Each state is a partial reasoning trace. You must:
        - Think step-by-step.
        - You must output exactly **one** next thought or action. Do not list multiple thoughts.
        - Each thought should advance reasoning, propose an idea, or invoke a tool call if beneficial.
        - Do NOT finalize an answer unless the state is clearly complete. 

        You have access to the following tools:

        1. equation_solver(input):
        - Solves symbolic algebraic equations (e.g., "x^2 - 4 = 0").
        - Use this for algebra.

        2. calculator(input):
        - Evaluates arithmetic expressions (e.g., "12 * 35 + 4.5").
        - Use this for all numerical calculations.

        3. wiki(input):
        - Searches Wikipedia for a summary of a specific topic.
        - Use this for general knowledge, definitions, or fact retrieval.
        - Input should be a specific search query (e.g., "Newton's Second Law").

        Example:
        TOOL_CALL:wiki:Barack Obama

        When using a tool, format your thought as:
        TOOL_CALL:<tool_name>:<input>

        Example:
        TOOL_CALL:calculator:2*pi + 3

        The system will run the tool and replace your call with TOOL_OUTPUT:<result> 
        in the next step's state.


        ---
        ### INSTRUCTIONS:
        1. **Thought**: Reason about the current state and what needs to be done next.
        2. **Action**: If you need a tool, output exactly: `TOOL_CALL:<tool_name>:<input>`
        3. **Observation**: The system will provide the `TOOL_OUTPUT`.
        4. **Answer**: When you have the solution, output `Answer: <final answer>`.
        
        - Do NOT generate the TOOL_OUTPUT yourself. Wait for the system.
        - Output strictly one step (Thought + Action) at a time. 

        ### OUTPUT FORMAT:
        Output Format:
        Always follow this format exactly:

        Plan:
        <Write a concise plan of how you will approach the problem, including any sub-steps or reasoning strategies.>

        Answer:
        <Provide the final answer to the query here. Do not include extra explanations. If a tool is used, you may incorporate its output into your answer.>

        ---

        ### GUIDELINES FOR GENERATING GOOD THOUGHTS:
        - Mix analytical, strategic, and computational ideas.
        - Consider whether a tool call would help reduce uncertainty or simplify algebra.
        - Avoid repeating the state verbatim; only add new progress.
        - Keep thoughts concise but meaningful.

        ---

        ### INPUT:
        State:
        {state}
    """
    return base_prompt

if __name__ == "__main__":
    datasets = [
        ("gsm8k", "main", "test"),
        ("EleutherAI/hendrycks_math", "algebra", "test"),
        ("cais/mmlu", "all", "validation"),
        ("hotpotqa/hotpot_qa", "distractor", "validation"),
        ("ai2_arc", "ARC-Easy", "test"),
        ("fireworks-ai/logiqa", None, "test")
    ]
    
    agent = ReActAgent(model="gemma-3-27b-it")
    results_summary = []
    
    for dataset_name, config, split in datasets:
        print(f"\nRunning on dataset: {dataset_name} (config={config})")
        try:
            # Fixed: only unpack one value
            res = agent.run_eval(dataset_name, config=config, split=split, max_samples=20)  
            print(f"Accuracy on {dataset_name} = {res['accuracy']:.4f}")
            results_summary.append({
                "dataset": dataset_name,
                "accuracy": res['accuracy'],
                "n_samples": res['n']
            })
        except Exception as e:
            print(f"Error running {dataset_name}: {e}")
            results_summary.append({
                "dataset": dataset_name,
                "accuracy": None,
                "n_samples": 0,
                "error": str(e)
            })

    # Save to CSV
    df = pd.DataFrame(results_summary)
    output_filename = "src/results/ReactAgent/experiment_results.csv"

    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_filename, index=False) 
    print(f"Results saved to {output_filename}")
    
    print("\n=== Summary ===")
    for r in results_summary:
        print(r)