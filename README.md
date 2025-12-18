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

## Results

We evaluated our approach against three baselines across five datasets. The table below reports the **Mean Accuracy ± Standard Deviation** and the **[95% Confidence Interval]** over three random seeds (30 samples per dataset).

| Dataset | I/O Baseline | CoT | ReAct | Proposed Approach |
| :--- | :--- | :--- | :--- | :--- |
| **GSM8K** | 0.933 ± 0.034 <br> [0.850, 1.000] | 0.947 ± 0.003 <br> [0.940, 0.955] | 0.900 ± 0.010 <br> [0.890, 0.910] | 0.856 ± 0.019 <br> [0.808, 0.903] |
| **HendrycksMath** | 0.978 ± 0.019 <br> [0.931, 1.000] | 0.925 ± 0.009 <br> [0.902, 0.948] | 0.820 ± 0.024 <br> [0.796, 0.844] | 0.844 ± 0.051 <br> [0.717, 0.972] |
| **AI2 ARC** | 0.967 ± 0.000 <br> [0.967, 0.967] | 0.915 ± 0.014 <br> [0.881, 0.950] | 0.900 ± 0.025 <br> [0.875, 0.925] | **1.000 ± 0.000** <br> **[1.000, 1.000]** |
| **HotpotQA** | 0.456 ± 0.102 <br> [0.202, 0.710] | 0.564 ± 0.028 <br> [0.494, 0.634] | 0.833 ± 0.041 <br> [0.792, 0.874] | 0.789 ± 0.019 <br> [0.741, 0.837] |
| **MMLU** | 0.744 ± 0.084 <br> [0.533, 0.955] | 0.828 ± 0.014 <br> [0.794, 0.862] | 0.800 ± 0.028 <br> [0.772, 0.828] | **1.000 ± 0.000** <br> **[1.000, 1.000]** |

### Key Findings
* **Knowledge Tasks:** Our method got all 30 samples right on AI2 ARC and MMLU, demonstrating superior handling of multi-choice and commonsense reasoning tasks compared to all baselines.
* **Math Tasks:** On simpler math tasks (GSM8K), our method slightly underperformed compared to CoT (0.856 vs 0.947), likely due to limited tool expressiveness and the high efficacy of linear reasoning for grade-school math.

## Authors
* **Abhijit Chunduru** - *UMass Amherst*
* **Aditi Ravindra** - *UMass Amherst*
* **Rishab Sharma** - *UMass Amherst*
