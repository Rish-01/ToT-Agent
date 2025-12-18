# ToT-Agent: From Thoughts to Trees

**Multi-Path Reasoning and Tool Integration in LLMs**

This repository contains the implementation of the **Multi-Path ReAct Agent**, a hybrid reasoning framework that extends the standard ReAct paradigm by integrating Tree-of-Thought (ToT) search strategies. By exploring multiple reasoning-action trajectories and systematically integrating external tools, this approach improves performance on complex reasoning tasks using smaller language models like Gemma 3 27b.

## Features

* **Multi-Path Exploration:** Unlike linear Chain-of-Thought (CoT) or standard ReAct, this agent explores a tree of reasoning steps, allowing it to backtrack and recover from early errors.
* **Tool Integration:** Integrates external tools, including an Equation Solver, Calculator, and Wikipedia Search, to provide factual grounding and reduce hallucinations.
* **Pruning & Evaluation:** Utilizes a "model-based evaluator" to score reasoning states (0-10) and prune unpromising paths using Breadth-First Search (BFS).

## Repository Structure

The source code is located in the `src/` directory:

* `src/proposed_method/our_method.py`: Implementation of the proposed Tree-based ReAct agent, including the BFS search algorithm, state expansion, and scoring mechanisms.
* `src/baselines/react_baseline.py`: Implementation of the standard single-path ReAct baseline, which interleaves thought and action steps linearly.
* `src/baselines/llm_baseline.py`: Script for running the Vanilla LLM (Input-Output) baseline.
* `src/baselines/cot_baseline` Script for Chain-of-Thought (CoT) baseline.
* `src/tools/dataset_utils.py`: Utilities for loading and processing benchmarks (GSM8K, MMLU, HotpotQA, AI2 ARC, Hendrycks MATH) from Hugging Face.
* `src/tools/math_tools.py`: Contains a script for math-based tools like a calculator and equation solver.
* `src/tools/retriever.py`: Contains a Wikipedia-based retriever tool.
  
## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rish-01/ToT-Agent.git
    cd ToT-Agent
    ```

2.  **Install dependencies:**
    Ensure you have the necessary Python packages installed (e.g., `datasets`, `google-generativeai` or `transformers`, `torch`).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    This project utilizes the **Gemma 3 27b-it** model via API. You must configure your API access accordingly.
    ```bash
    export GEMMA_API_KEY="your_api_key_here"
    ```

## Usage

Run the proposed approach using:
```bash
  python -m src.proposed_method.our_method
```
