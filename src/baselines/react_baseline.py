import os
import re
from google import genai
from datasets import load_dataset

from src.tools.math_tools import EquationSolverTool, CalculatorTool

class ReActAgent:
    def __init__(self, model="gemma-3-27b-it"):
        api_key = os.environ["GEMMA_API_KEY"]
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def react_step(self, prompt, tools=None, max_steps=5):
        # state = prompt
        tools = tools or {}

        tool_descriptions = "\n".join(
            [f"{name}: {tool.__doc__}" for name, tool in tools.items()]
        )
        state = f"You are a ReAct AI agent that can solve problems using the following tools:\n{tool_descriptions}\n"
        state += f"Use the tools to help you answer the question step-by-step.\n"
        state += "When you need to use a tool, respond in this format: <ToolName>: <input>\n"
        state += f"You have {max_steps} iterations to answer the question. You can reason using tools in the initial steps and answer on the last step.\n"
        state += "Always provide a final answer using the format: Answer: <your final answer>\n\n"
        state += f"Problem:\n{prompt}"

        for i in range(max_steps):
            state += f"\n\nYou are currently on Step {i+1}:\n"
            resp = self.client.models.generate_content(
                model=self.model,
                contents=state
            )
            output = resp.text

            tool_pattern = re.compile(r"<(\w+)>:\s*(.+)")

            matches = tool_pattern.findall(output)
            for match in matches:
                tool_name, tool_input = match
                tool_name = tool_name.lower()  
                if tool_name in tools:
                    tool_result = tools[tool_name].run(tool_input)
                    # Append tool output to state for next step
                    state += f"{output}\n" + f"Tool Output: {tool_result}\n"
                else:
                    state += f"{output}\n" + f"ERROR: Tool '{tool_name}' not found.\n" 

            if "Answer:" in output:
                return output, state 
            
        return output, state


    def run_on_dataset(self, dataset_name, max_samples=None, tools=None, split="test", config="main"):
        dataset = load_dataset(dataset_name, config, split=split)
        if max_samples:
            dataset = dataset.select(range(max_samples))
        
        results = []
        for item in dataset:
            prompt = self.get_prompt(item)
            print("Prompt:", prompt)
            output, state = self.react_step(prompt, tools, max_steps=5)
            print("Reasoning Trace:", state)
            results.append(self.evaluate(item, state))
        return results
    
    def get_prompt(self, item):
        return item.get("question", "")
    
    def evaluate(self, item, output):
        matches = re.findall(r"Answer:\s*(.*)", output)
        if matches:
            pred_answer = matches[-1].strip()
        else:
            return {"correct": 0}
        
        # Extract reference answer
        ref_answer = item.get("answer", "").strip()
        
        # Extract numeric part from reference answer
        numbers = re.findall(r"\d+", ref_answer)
        if numbers:
            ref_number = numbers[-1] 
        else:
            ref_number = None

        try:
            correct = float(pred_answer) == float(ref_number)
        except:
            correct = False

        return {"correct": int(correct)}


if __name__ == "__main__":
    # Test ReActAgent on gsm8k
    agent = ReActAgent(model="gemma-3-27b-it")

    tools = {
        "calculator": CalculatorTool(),
        "equation_solver": EquationSolverTool()
    }
    results = agent.run_on_dataset("gsm8k", max_samples=1, tools=tools, config="main")
    correct = sum([r['correct'] for r in results])
    print(f"Accuracy: {correct/len(results):.2f}")