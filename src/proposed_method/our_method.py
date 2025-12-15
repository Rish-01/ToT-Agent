import os
import re
import json
import time
import pandas as pd

from google import genai
from datasets import load_dataset

from src.tools.dataset_utils import DatasetUtils
from src.tools.math_tools import EquationSolverTool, CalculatorTool
from src.tools.retriever import WikipediaTool

from collections import deque
from typing import List, Dict

class TreeNode:
    """
    A node in the Tree of Thoughts structure.
    Reference: https://huggingface.co/blog/sadhaklal/tree-of-thoughts
    """
    def __init__(self, state: str, thought: str, value: float = None):
        self.state = state
        self.thought = thought
        self.value = value
        self.children = []

class ToTAgent:
    def __init__(self, model="gemma-3-27b-it"):
        api_key = os.environ["GEMMA_API_KEY"]
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.root = TreeNode(state='', thought='')
        self.breadth_limit = 1
        self.n_steps = 5
        self.num_samples = 2

        self.tools = {
            "equation_solver": EquationSolverTool(),
            "calculator": CalculatorTool(),
            "wiki": WikipediaTool()
        }

    def tot_step(self, node: TreeNode, state: str, num_samples: int) -> List[str]:
        prompt = get_base_prompt(state)
        samples = []
        for _ in range(num_samples):
            # While loop to retry on rate limit errors
            while True:
                try:
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config={
                            "temperature": 0.6,
                            "top_p": 0.9,
                            "top_k": 40,
                        }
                    )
                    thought = resp.text

                    # Process tool calls in the thought
                    final_thought = self.resolve_tool_calls(thought)
                    samples.append(final_thought)
                    break  
                
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                        print(f"Rate limit hit! Waiting 60 seconds...")
                        time.sleep(60)
                        print("Resuming...")
                    else:
                        raise

        return samples

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

    def evaluate_state(self, states: List[str]) -> List[float]:
        eval_prompt = evaluation_prompt(states)
        
        while True:
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=eval_prompt,
                    config={
                        "temperature": 0.0,
                        "top_p": 0.9,
                        "top_k": 40,
                    }
                )
                response = resp.text
                break  

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    print(f"Rate limit hit! Waiting 60 seconds...")
                    time.sleep(60)
                    print("Resuming...")
                else:
                    # Non-quota error, raise it
                    raise

        match = re.search(r'\{.*\}', response, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")

        data = json.loads(match.group(0))
        print(data)
        return [data[str(i+1)] for i in range(len(states))]


    def bfs(self, initial_prompt: str, output_treeviz_file: str) -> str:
        # Reset root with the initial prompt
        self.root = TreeNode(state=initial_prompt, thought='')
        queue = deque([self.root])
        
        for step in range(1, self.n_steps + 1):
            level_size = len(queue)
            next_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                # generate thoughts 
                thoughts = self.tot_step(node, state=node.state, num_samples=self.num_samples)
                # updated states
                if node.state == "":
                    updated_states = thoughts
                else:
                    updated_states = [node.state + "\n" + t for t in thoughts]
                # create children nodes 
                for state, thought in zip(updated_states, thoughts):
                    child = TreeNode(state=state, thought=thought)
                    node.children.append(child)
                    next_level.append(child)
                    
                    # Check if this thought contains a final answer
                    if self.has_final_answer(thought):
                        child.is_terminal = True
                        os.makedirs(os.path.dirname(output_treeviz_file), exist_ok=True)
                        self.visualize_tree_html(self.root, output_treeviz_file)
                        return child.state
            
            # Create a new queue for the next level
            queue = deque(next_level)
            
            # Score all nodes in the current level
            states = [node.state for node in queue]
            values = self.evaluate_state(states=states)
            for node, value in zip(queue, values):
                node.value = value
            
            # Prune nodes based on value
            sorted_nodes = sorted(queue, key=lambda n: n.value, reverse=True)
            if step == self.n_steps:
                # Get the best node only at the last step
                top_nodes = sorted_nodes[:1]       
            else:
                # Get top-k nodes
                top_nodes = sorted_nodes[:self.breadth_limit]
            queue = deque(top_nodes)
        
        os.makedirs(os.path.dirname(output_treeviz_file), exist_ok=True)
        self.visualize_tree_html(self.root, output_treeviz_file)
        # return the thought of the best node at the end
        best = queue[0]
        return best.state

    def has_final_answer(self, thought: str) -> bool:
        """Check if a thought contains a final answer"""
        # Look for answer pattern in the thought
        thought_lower = thought.lower()
        
        # Check if Answer section has content after it
        if "answer:" in thought_lower:
            answer_section = thought.split("Answer:", 1)[-1].strip()
            # Check if there's actual content (not empty, not placeholder)
            if answer_section and len(answer_section) > 10:
                # Avoid false positives from placeholders
                placeholders = ["<provide", "to be determined", "pending", "n/a", "tbd"]
                if not any(p in answer_section.lower() for p in placeholders):
                    return True
        
        return False

    def run_eval(self, dataset_name, config=None, split="test", max_samples=None, seed=42):
        ds = DatasetUtils.load_and_prep_dataset(dataset_name, config, split, max_samples, seed)
        evaluator = DatasetUtils.get_evaluator(dataset_name)
        
        if evaluator is None:
            raise ValueError(f"No evaluator found for dataset: {dataset_name}")
        
        results = []
        
        # Create log directory
        log_dir = f"src/results/ToTAgent/logs/{dataset_name.replace('/', '_')}"
        os.makedirs(log_dir, exist_ok=True)
        
        for i, item in enumerate(ds):
            prompt = DatasetUtils.make_prompt(dataset_name, item)
            # Ensure directory exists for visualization
            viz_dir = f"tree_viz/{dataset_name.replace('/', '_')}_seed_{seed}"
            os.makedirs(viz_dir, exist_ok=True)
            output_file = f"{viz_dir}/sample_{i}.html"
            
            try:
                print(f"\nProcessing Sample {i+1}...")
                final_state = self.bfs(prompt, output_file)
                prediction = final_state 
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                prediction = ""
            
            is_correct = evaluator(item, prediction)
            print(f"Correct: {is_correct}")
            
            # Store result with additional context
            result_entry = {
                "index": i,
                "correct": is_correct,
                "prediction": prediction,
                "tree_viz_file": output_file
            }
            
            # Add ground truth answer based on dataset
            if "gsm8k" in dataset_name.lower():
                result_entry["ground_truth"] = item.get("answer", "")
                result_entry["question"] = item.get("question", "")
            elif "math" in dataset_name.lower() or "hendrycks" in dataset_name.lower():
                result_entry["ground_truth"] = item.get("solution", item.get("answer", ""))
                result_entry["problem"] = item.get("problem", "")
            elif "hotpot" in dataset_name.lower():
                result_entry["ground_truth"] = item.get("answer", "")
                result_entry["question"] = item.get("question", "")
            elif "arc" in dataset_name.lower():
                result_entry["ground_truth"] = item.get("answerKey", "")
                result_entry["question"] = item.get("question", "")
            elif "mmlu" in dataset_name.lower():
                result_entry["ground_truth"] = item.get("answer", "")
                result_entry["question"] = item.get("question", "")
            elif "logiqa" in dataset_name.lower():
                result_entry["ground_truth"] = item.get("answer", item.get("label", ""))
                result_entry["question"] = item.get("question", "")
            
            results.append(result_entry)
            
            # Log every 20 samples
            if (i + 1) % 20 == 0:
                interim_acc = sum(r["correct"] for r in results) / len(results)
                print(f"\n[CHECKPOINT] Samples processed: {i+1}/{len(ds)} | Interim Accuracy: {interim_acc:.4f}")
                
                # Save checkpoint results to file
                checkpoint_file = f"{log_dir}/checkpoint_sample_{i+1}_seed_{seed}.json"
                checkpoint_data = {
                    "dataset": dataset_name,
                    "samples_processed": i + 1,
                    "interim_accuracy": interim_acc,
                    "seed": seed,
                    "results": results
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"Checkpoint saved to {checkpoint_file}")
        
        # Calculate final metrics
        accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
        
        # Save wrong answers to separate file for analysis
        wrong_answers = [r for r in results if not r["correct"]]
        if wrong_answers:
            wrong_file = f"{log_dir}/wrong_answers_seed_{seed}.json"
            with open(wrong_file, 'w') as f:
                json.dump({
                    "dataset": dataset_name,
                    "seed": seed,
                    "total_samples": len(results),
                    "num_wrong": len(wrong_answers),
                    "accuracy": accuracy,
                    "wrong_answers": wrong_answers
                }, f, indent=2)
            print(f"\nWrong answers saved to {wrong_file}")
        
        # Rename tree viz files to include correctness
        for r in results:
            old_path = r["tree_viz_file"]
            if os.path.exists(old_path):
                # Add _correct or _wrong prefix to filename
                dirname = os.path.dirname(old_path)
                filename = os.path.basename(old_path)
                status = "correct" if r["correct"] else "wrong"
                new_filename = f"{status}_{filename}"
                new_path = os.path.join(dirname, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    r["tree_viz_file"] = new_path
                except Exception as e:
                    print(f"Warning: Could not rename {old_path}: {e}")
        
        # Create the summary dictionary your loop expects
        summary_res = {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "n": len(results),
            "results": results
        }
        
        # Return accuracy AND the dictionary
        return accuracy, summary_res

    
    def _tree_to_dict(self, node: TreeNode) -> Dict:
        """Convert TreeNode to dictionary for JSON serialization."""
        return {
            "name": node.thought[:100] + "..." if len(node.thought) > 100 else node.thought,
            "full_thought": node.thought,
            "state": node.state[:200] + "..." if len(node.state) > 200 else node.state,
            "value": node.value,
            "children": [self._tree_to_dict(child) for child in node.children]
        }

    def visualize_tree_html(self, root: TreeNode, output_file: str):
        import json
        import uuid  # Required for unique node identification
        
        tree_data = self._tree_to_dict(root)
        
        def preprocess_node(node):
            # Generate a unique ID for D3 data binding
            node_id = str(uuid.uuid4())
            node["id"] = node_id
            
            full_thought = node.get("full_thought", "")
            state = node.get("state", "")
            
            # HTML-safe strings for the sidebar (replacing newlines)
            node["html_thought"] = full_thought.replace("\n", "<br>")
            node["html_state"] = state.replace("\n", "<br>")
            
            # Graph Label: We append a tiny slice of the ID (e.g. #a1b2) 
            # to VISUALLY guarantee that nodes are distinct, even if text is identical.
            clean_thought = full_thought.replace("\n", " ")
            _id = node_id[:4]
            if len(clean_thought) > 50:
                node["graph_label"] = f"{clean_thought[:50]}... [#{node_id}]"
            else:
                node["graph_label"] = f"{clean_thought} [#{node_id}]"
                
            # Process children
            for child in node.get("children", []):
                preprocess_node(child)

        preprocess_node(tree_data)

        # 3. Generate HTML
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Tree of Thoughts</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body { margin: 0; height: 100vh; display: flex; font-family: sans-serif; background: #f8f9fa; }
                #tree-container { flex: 1; height: 100%; overflow: auto; position: relative; }
                #sidebar { width: 400px; height: 100%; background: white; border-left: 1px solid #ddd; padding: 20px; box-sizing: border-box; overflow-y: auto; box-shadow: -2px 0 10px rgba(0,0,0,0.05); }
                
                .node circle { fill: #fff; stroke: #555; stroke-width: 2px; cursor: pointer; transition: all 0.3s; }
                .node circle:hover { stroke-width: 4px; }
                .node circle.selected { stroke: #d63384; stroke-width: 4px; fill: #fff0f6; }
                .node text { font: 12px sans-serif; pointer-events: none; }
                .link { fill: none; stroke: #ccc; stroke-width: 1.5px; }
                
                h2 { margin-top: 0; color: #333; }
                .content-box { background: #f1f3f5; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 0.9rem; white-space: pre-wrap; word-wrap: break-word; margin-bottom: 20px; }
                .score-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; color: white; font-weight: bold; margin-bottom: 10px;}
            </style>
        </head>
        <body>
            <div id="tree-container"></div>
            <div id="sidebar">
                <h2>Node Inspector</h2>
                <div id="inspector-content">
                    <p style="color: #888; font-style: italic;">Click on a node to view full details.</p>
                </div>
            </div>

            <script>
                const treeData = """ + json.dumps(tree_data) + """;
                
                const width = 2000;
                const height = 1000;
                const margin = {top: 20, right: 150, bottom: 20, left: 100};

                const svg = d3.select("#tree-container").append("svg")
                    .attr("width", width)
                    .attr("height", height)
                    .append("g")
                    .attr("transform", `translate(${margin.left},${margin.top})`);

                const tree = d3.tree().size([height - margin.top - margin.bottom, width - margin.left - margin.right]);
                const root = d3.hierarchy(treeData);
                tree(root);

                // Draw Links
                svg.selectAll(".link")
                    .data(root.links())
                    .enter().append("path")
                    .attr("class", "link")
                    .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));

                // ---------------------------------------------------------
                // FIX: Key Function added to .data()
                // We use 'd.data.id' to ensure D3 distinguishes every node correctly.
                // ---------------------------------------------------------
                const node = svg.selectAll(".node")
                    .data(root.descendants(), d => d.data.id) 
                    .enter().append("g")
                    .attr("class", "node")
                    .attr("transform", d => `translate(${d.y},${d.x})`);

                node.append("circle")
                    .attr("r", 8)
                    .attr("stroke", d => {
                        const val = d.data.value;
                        if (val >= 8) return "#28a745"; 
                        if (val >= 5) return "#ffc107"; 
                        return "#dc3545"; 
                    })
                    .on("click", function(event, d) {
                        d3.selectAll(".node circle").classed("selected", false);
                        d3.select(this).classed("selected", true);
                        
                        const val = d.data.value !== null ? d.data.value.toFixed(2) : "N/A";
                        const color = (d.data.value >= 8) ? "#28a745" : (d.data.value >= 5 ? "#e0a800" : "#dc3545");
                        
                        document.getElementById("inspector-content").innerHTML = `
                            <h3>Evaluation Score</h3>
                            <div class="score-badge" style="background-color: ${color}">${val}</div>
                            <h3>Unique Node ID</h3>
                            <div style="font-family:monospace; margin-bottom:10px; color:#666;">${d.data.id}</div>
                            <h3>Reasoning (Thought)</h3>
                            <div class="content-box">${d.data.html_thought}</div>
                            <h3>State Snapshot</h3>
                            <div class="content-box" style="font-size: 0.8rem; color: #555;">${d.data.html_state}</div>
                        `;
                    });

                node.append("text")
                    .attr("dy", "0.31em")
                    .attr("x", d => d.children ? -12 : 12)
                    .style("text-anchor", d => d.children ? "end" : "start")
                    .text(d => {
                        const val = d.data.value !== null ? `(${d.data.value})` : "";
                        return `${d.data.graph_label} ${val}`;
                    });
            </script>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"Visualization saved to {output_file}")


def get_base_prompt(state: str) -> str:
    base_prompt = f"""
        You are a Tree-of-Thought (ToT) reasoning agent.
        Your job is to expand reasoning states into multiple possible next thoughts. 
        Each state is a partial reasoning trace. You must:
        - Think step-by-step.
        - You must output exactly **one** next thought or action. Do not list multiple thoughts.
        - Each thought should advance reasoning, propose an idea, or invoke a tool call if beneficial.
        - **IMPORTANT**: Only provide a final answer in the "Answer:" section when you have definitively solved the problem. If you're still reasoning, leave the Answer section empty or write "Pending further analysis".
        - Try to be as concise as possible while still being meaningful.

        You also have access to the following tools:
        1. equation_solver(input):
        - Solves symbolic algebraic equations of the form "expression = expression".
        - Use when the task requires algebraic manipulation or solving for variables.

        2. calculator(input):
        - Evaluates arithmetic or numerical expressions (supports pi, e).
        - Use for numeric calculations.

        3. wiki(input):
        - Searches Wikipedia for a summary of a specific topic.
        - Use this for general knowledge, definitions, or fact retrieval.
        - Input should be a specific search query (e.g., "Newton's Second Law").

        When using a tool, format your thought as:
        TOOL_CALL:<tool_name>:<input>

        Example:
        TOOL_CALL:calculator:2*pi + 3

        Example:
        TOOL_CALL:wiki:Barack Obama

        The system will run the tool and replace your call with TOOL_OUTPUT:<result> 
        in the next step's state.

        ---
        ### OUTPUT FORMAT:
        Always follow this format exactly:

        Plan:
        <Write a concise plan of how you will approach the problem, including any sub-steps or reasoning strategies.>

        Action:
        <Based on your plan, decide which tools to call and in what order. If no tool is needed, describe the reasoning steps you will take. Use the tool call format TOOL_CALL:<tool_name>:<input>>

        Answer:
        <**Only fill this if you have the complete final answer.** Otherwise, write "Pending" or leave instructions for next steps.>

        ---
        ### GUIDELINES FOR GENERATING GOOD THOUGHTS:
        - Mix analytical, strategic, and computational ideas.
        - Consider whether a tool call would help reduce uncertainty or simplify algebra.
        - Avoid repeating the state verbatim; only add new progress.
        - Keep thoughts concise but meaningful.
        - **Signal completion clearly**: When you have the final answer, state it explicitly in the Answer section.

        ---
        ### INPUT:
        State:
        {state}
        """
    return base_prompt


def evaluation_prompt(states: List[str]) -> str:
    states_block = "\n\t".join(f"State {i}: {s}" for i, s in enumerate(states, start=1))

    eval_prompt = f"""
        You are evaluating intermediate reasoning states.

        Goal:
        Assess how promising each state is for eventually reaching a correct and high-quality final solution.

        Instructions:
        - You will be given a numbered list of states.
        - For each state, assign a score from 0 to 10.
        - A higher score means the state is more coherent, logically consistent, closer to a good solution, and more promising to expand further.
        - A lower score means the state is confused, inconsistent, redundant, or unlikely to lead to a correct solution.
        - Score each state independently.
        - Output ONLY a JSON dictionary where keys are state indices (as strings) and values are numbers.
        - Output only a JSON dictionary. Do not include any text outside the JSON object.
        - Your response must begin with '{{' and end with '}}'.

        States:
        {states_block}

        Output format:
        {{
            "1": score_for_state_1,
            "2": score_for_state_2,
            ...
        }}
    """
    return eval_prompt

if __name__ == "__main__":
    datasets = [
        ("gsm8k", "main", "test"),
        ("EleutherAI/hendrycks_math", "algebra", "test"),
        ("cais/mmlu", "all", "validation"),
        ("hotpotqa/hotpot_qa", "distractor", "validation"),
        ("ai2_arc", "ARC-Easy", "test"),
        ("fireworks-ai/logiqa", None, "test")
    ]
    agent = ToTAgent(model="gemma-3-27b-it")
    results_summary = []
    
    # Setup output file
    output_filename = "src/results/ToTAgent/experiment_results.csv"
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name, config, split in datasets:
        for run in range(3):  
            seed = [42, 43, 44][run]
            print(f"\nRunning on dataset: {dataset_name} (config={config}, seed={seed})")
            try:
                acc, res = agent.run_eval(dataset_name, config=config, split=split, max_samples=30, seed=seed)  
                print(f"Accuracy on {dataset_name} = {res['accuracy']:.4f}")
                results_summary.append({
                    "dataset": dataset_name,
                    "config": config,
                    "seed": seed,
                    "run": run + 1,
                    "accuracy": res['accuracy'],
                    "n_samples": res['n']
                })
            except Exception as e:
                print(f"Error running {dataset_name}: {e}")
                results_summary.append({
                    "dataset": dataset_name,
                    "config": config,
                    "seed": seed,
                    "run": run + 1,
                    "accuracy": None,
                    "n_samples": 0,
                    "error": str(e)
                })
            
            # Save results after each evaluation
            df = pd.DataFrame(results_summary)
            df.to_csv(output_filename, index=False)
            print(f"✓ Results updated in {output_filename}")
    
    print(f"\n=== Final Results saved to {output_filename} ===")
    print("\n=== Summary ===")
    for r in results_summary:
        print(r)