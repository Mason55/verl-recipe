# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SWE Agent Dataset Generator

Supports two data sources:
1. Simple test cases (for quick validation)
2. SWE-bench Lite (for full evaluation)

Data format is VERL-compatible:
- prompt: Conversation format containing problem_statement
- reward_model: Evaluation configuration
- extra_info: Contains repo_content, expected_patch, etc.
- agent_name: "swe_agent"
"""

import argparse
import json
import os
from typing import Optional

import pandas as pd

# SWE-Agent system prompt
SWE_AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer"
    " to solve software engineering tasks.\n\n"
    "When working on a task:\n"
    "1. First understand the problem by reading relevant code\n"
    "2. Make minimal changes to fix the issue\n"
    "3. Verify your changes work correctly\n\n"
    "Your goal is to create a patch that resolves the issue"
    " described in the problem statement."
)


def create_swe_prompt(problem_statement: str, repo_info: Optional[dict] = None) -> list[dict[str, str]]:
    """Create SWE-Agent prompt format.

    Args:
        problem_statement: Problem description.
        repo_info: Optional repository information.

    Returns:
        Prompt in conversation format.
    """
    user_content = f"""<problem_statement>
{problem_statement}
</problem_statement>

Please analyze the problem and implement the necessary changes to resolve it.
Make minimal changes to the codebase while ensuring the issue is fully addressed."""

    if repo_info:
        repo_name = repo_info.get("repo_name", "unknown")
        base_commit = repo_info.get("base_commit", "")
        user_content = f"""Repository: {repo_name}
Base commit: {base_commit}

{user_content}"""

    return [
        {"role": "system", "content": SWE_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_simple_test_data(num_samples: int, split: str, agent_name: str = "swe_agent") -> pd.DataFrame:
    """Generate simple test data (for quick validation).

    Args:
        num_samples: Number of samples.
        split: Dataset split (train/test).
        agent_name: Agent name.

    Returns:
        Dataset in DataFrame format.
    """
    rl_dataset = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
        "agent_name": [],
    }

    # Diverse test cases covering different SWE skills
    test_cases = [
        # --- File operations ---
        {
            "problem_statement": "rename 1.txt to 2.txt",
            "repo_content": {"1.txt": "Hello World"},
            "expected_patch": """diff --git a/1.txt b/2.txt
similarity index 100%
rename from 1.txt
rename to 2.txt""",
        },
        {
            "problem_statement": "Create a new file called hello.py that prints 'Hello, World!'",
            "repo_content": {},
            "expected_patch": """diff --git a/hello.py b/hello.py
new file mode 100644
--- /dev/null
+++ b/hello.py
@@ -0,0 +1 @@
+print('Hello, World!')""",
        },
        # --- Bug fixes ---
        {
            "problem_statement": "Fix the bug in calculator.py: the add function should return a + b, not a - b",
            "repo_content": {"calculator.py": "def add(a, b):\n    return a - b"},
            "expected_patch": """diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b""",
        },
        {
            "problem_statement": "Fix the off-by-one error in range_sum.py: the function should include the end value",
            "repo_content": {"range_sum.py": "def range_sum(start, end):\n    return sum(range(start, end))"},
            "expected_patch": """diff --git a/range_sum.py b/range_sum.py
--- a/range_sum.py
+++ b/range_sum.py
@@ -1,2 +1,2 @@
 def range_sum(start, end):
-    return sum(range(start, end))
+    return sum(range(start, end + 1))""",
        },
        {
            "problem_statement": "Fix the TypeError in greet.py: the function should handle None name by using 'World' as default",
            "repo_content": {"greet.py": "def greet(name):\n    return 'Hello, ' + name + '!'"},
            "expected_patch": """diff --git a/greet.py b/greet.py
--- a/greet.py
+++ b/greet.py
@@ -1,2 +1,3 @@
-def greet(name):
+def greet(name=None):
+    name = name or 'World'
     return 'Hello, ' + name + '!'""",
        },
        # --- Add functionality ---
        {
            "problem_statement": "Add a multiply function to math_ops.py that multiplies two numbers",
            "repo_content": {"math_ops.py": "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n"},
            "expected_patch": """diff --git a/math_ops.py b/math_ops.py
--- a/math_ops.py
+++ b/math_ops.py
@@ -3,0 +4,3 @@
+def multiply(a, b):
+    return a * b
+""",
        },
        {
            "problem_statement": "Add a __str__ method to the Person class in person.py that returns 'Name: {name}, Age: {age}'",
            "repo_content": {"person.py": "class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n"},
            "expected_patch": """diff --git a/person.py b/person.py
--- a/person.py
+++ b/person.py
@@ -3,0 +4,3 @@
+    def __str__(self):
+        return f'Name: {self.name}, Age: {self.age}'
+""",
        },
        {
            "problem_statement": "Add input validation to divide.py: raise ValueError when divisor is zero",
            "repo_content": {"divide.py": "def divide(a, b):\n    return a / b\n"},
            "expected_patch": """diff --git a/divide.py b/divide.py
--- a/divide.py
+++ b/divide.py
@@ -1,2 +1,4 @@
 def divide(a, b):
+    if b == 0:
+        raise ValueError('Cannot divide by zero')
     return a / b""",
        },
        # --- Refactoring ---
        {
            "problem_statement": "Refactor utils.py: rename the function 'calc' to 'calculate_total' for better readability",
            "repo_content": {"utils.py": "def calc(items):\n    total = 0\n    for item in items:\n        total += item\n    return total\n"},
            "expected_patch": """diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
@@ -1,5 +1,5 @@
-def calc(items):
+def calculate_total(items):
     total = 0
     for item in items:
         total += item
     return total""",
        },
        {
            "problem_statement": "Replace the manual loop in stats.py with a list comprehension",
            "repo_content": {"stats.py": "def get_even_numbers(numbers):\n    result = []\n    for n in numbers:\n        if n % 2 == 0:\n            result.append(n)\n    return result\n"},
            "expected_patch": """diff --git a/stats.py b/stats.py
--- a/stats.py
+++ b/stats.py
@@ -1,6 +1,2 @@
 def get_even_numbers(numbers):
-    result = []
-    for n in numbers:
-        if n % 2 == 0:
-            result.append(n)
-    return result
+    return [n for n in numbers if n % 2 == 0]""",
        },
        # --- Config / text changes ---
        {
            "problem_statement": "Update config.py: change the DEFAULT_PORT from 8080 to 9090",
            "repo_content": {"config.py": "DEFAULT_PORT = 8080\nDEBUG = False\nMAX_RETRIES = 3\n"},
            "expected_patch": """diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1,3 +1,3 @@
-DEFAULT_PORT = 8080
+DEFAULT_PORT = 9090
 DEBUG = False
 MAX_RETRIES = 3""",
        },
        {
            "problem_statement": "Add a .gitignore file that ignores __pycache__, *.pyc, and .env files",
            "repo_content": {"main.py": "print('hello')\n"},
            "expected_patch": """diff --git a/.gitignore b/.gitignore
new file mode 100644
--- /dev/null
+++ b/.gitignore
@@ -0,0 +1,3 @@
+__pycache__/
+*.pyc
+.env""",
        },
        # --- Multi-file awareness ---
        {
            "problem_statement": "Fix the import in main.py: it imports 'helper' but the function in utils.py is called 'help_func'. Rename the function in utils.py to 'helper'.",
            "repo_content": {
                "main.py": "from utils import helper\n\nprint(helper())\n",
                "utils.py": "def help_func():\n    return 'I am helping'\n",
            },
            "expected_patch": """diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
@@ -1,2 +1,2 @@
-def help_func():
+def helper():
     return 'I am helping'""",
        },
        {
            "problem_statement": "Add a README.md file that describes the project as 'A simple Python calculator' with usage instructions showing 'python calculator.py'",
            "repo_content": {"calculator.py": "def add(a, b):\n    return a + b\n\nif __name__ == '__main__':\n    print(add(1, 2))\n"},
            "expected_patch": """diff --git a/README.md b/README.md
new file mode 100644
--- /dev/null
+++ b/README.md
@@ -0,0 +1,5 @@
+# Calculator
+
+A simple Python calculator.
+
+Usage: `python calculator.py`""",
        },
        # --- Error handling ---
        {
            "problem_statement": "Add try-except error handling to file_reader.py: catch FileNotFoundError and print 'File not found' instead of crashing",
            "repo_content": {"file_reader.py": "def read_file(path):\n    with open(path) as f:\n        return f.read()\n"},
            "expected_patch": """diff --git a/file_reader.py b/file_reader.py
--- a/file_reader.py
+++ b/file_reader.py
@@ -1,3 +1,6 @@
 def read_file(path):
-    with open(path) as f:
-        return f.read()
+    try:
+        with open(path) as f:
+            return f.read()
+    except FileNotFoundError:
+        print('File not found')
+        return None""",
        },
        {
            "problem_statement": "Fix the logical error in is_palindrome.py: the function should be case-insensitive",
            "repo_content": {"is_palindrome.py": "def is_palindrome(s):\n    return s == s[::-1]\n"},
            "expected_patch": """diff --git a/is_palindrome.py b/is_palindrome.py
--- a/is_palindrome.py
+++ b/is_palindrome.py
@@ -1,2 +1,3 @@
 def is_palindrome(s):
-    return s == s[::-1]
+    s = s.lower()
+    return s == s[::-1]""",
        },
    ]

    for idx in range(num_samples):
        case = test_cases[idx % len(test_cases)]

        # Create prompt
        prompt = create_swe_prompt(case["problem_statement"])

        # Reward model config
        reward_model = {
            "style": "swe_agent",
            "ground_truth": case["expected_patch"],
        }

        # Extra info
        extra_info = {
            "index": idx,
            "split": split,
            "repo_content": case["repo_content"],
            "expected_patch": case["expected_patch"],
            "problem_statement": case["problem_statement"],
        }

        rl_dataset["prompt"].append(prompt)
        rl_dataset["data_source"].append("swe_agent_simple")
        rl_dataset["ability"].append("software_engineering")
        rl_dataset["reward_model"].append(reward_model)
        rl_dataset["extra_info"].append(extra_info)
        rl_dataset["agent_name"].append(agent_name)

    return pd.DataFrame(data=rl_dataset)


def load_swebench_lite(swebench_path: str, split: str, agent_name: str = "swe_agent") -> pd.DataFrame:
    """Load SWE-bench Lite dataset.

    Args:
        swebench_path: Path to SWE-bench data file.
        split: Dataset split.
        agent_name: Agent name.

    Returns:
        Dataset in DataFrame format.
    """
    rl_dataset = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
        "agent_name": [],
    }

    # Read SWE-bench data
    if swebench_path.endswith(".json"):
        with open(swebench_path) as f:
            data = json.load(f)
    elif swebench_path.endswith(".jsonl"):
        data = []
        with open(swebench_path) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {swebench_path}")

    for idx, item in enumerate(data):
        instance_id = item.get("instance_id", f"instance_{idx}")
        problem_statement = item.get("problem_statement", "")

        # Repository info
        repo_info = {
            "repo_name": item.get("repo", ""),
            "base_commit": item.get("base_commit", ""),
        }

        # Create prompt
        prompt = create_swe_prompt(problem_statement, repo_info)

        # Reward model config
        reward_model = {
            "style": "swe_bench",
            "instance_id": instance_id,
            "test_patch": item.get("test_patch", ""),
            "gold_patch": item.get("patch", ""),
        }

        # Extra info
        extra_info = {
            "index": idx,
            "split": split,
            "instance_id": instance_id,
            "repo": item.get("repo", ""),
            "base_commit": item.get("base_commit", ""),
            "problem_statement": problem_statement,
            "hints_text": item.get("hints_text", ""),
            "created_at": item.get("created_at", ""),
            "version": item.get("version", ""),
            "FAIL_TO_PASS": item.get("FAIL_TO_PASS", ""),
            "PASS_TO_PASS": item.get("PASS_TO_PASS", ""),
        }

        rl_dataset["prompt"].append(prompt)
        rl_dataset["data_source"].append("swe_bench_lite")
        rl_dataset["ability"].append("software_engineering")
        rl_dataset["reward_model"].append(reward_model)
        rl_dataset["extra_info"].append(extra_info)
        rl_dataset["agent_name"].append(agent_name)

    return pd.DataFrame(data=rl_dataset)


def main():
    parser = argparse.ArgumentParser(description="SWE Agent Dataset Generator")
    parser.add_argument(
        "--mode",
        choices=["simple", "swebench"],
        default="simple",
        help="Data generation mode: 'simple' for test cases, 'swebench' for SWE-bench Lite",
    )
    parser.add_argument("--train_size", type=int, default=100, help="Number of training samples (for simple mode)")
    parser.add_argument("--test_size", type=int, default=10, help="Number of testing samples (for simple mode)")
    parser.add_argument("--swebench_train", type=str, default=None, help="Path to SWE-bench train data")
    parser.add_argument("--swebench_test", type=str, default=None, help="Path to SWE-bench test data")
    parser.add_argument("--output_dir", default="data/swe_agent", help="Directory to save the dataset")
    parser.add_argument("--agent_name", default="swe_agent", help="Name of the agent")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Check pyarrow
    try:
        import importlib.util

        if importlib.util.find_spec("pyarrow") is None:
            raise ImportError("pyarrow not found")
    except ImportError as err:
        raise ImportError(
            "pyarrow is required for parquet support. Please install it with: pip install pyarrow"
        ) from err

    if args.mode == "simple":
        # Generate simple test data
        print("Generating simple test data...")
        train_dataset = generate_simple_test_data(args.train_size, "train", args.agent_name)
        test_dataset = generate_simple_test_data(args.test_size, "test", args.agent_name)
    else:
        # Load SWE-bench data
        print("Loading SWE-bench Lite data...")
        if args.swebench_train is None or args.swebench_test is None:
            raise ValueError("--swebench_train and --swebench_test are required for swebench mode")
        train_dataset = load_swebench_lite(args.swebench_train, "train", args.agent_name)
        test_dataset = load_swebench_lite(args.swebench_test, "test", args.agent_name)

    # Save dataset
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print("\nDataset generation completed!")
    print(f"Train dataset: {len(train_dataset)} samples -> {train_path}")
    print(f"Test dataset: {len(test_dataset)} samples -> {test_path}")

    # Print dataset sample
    print("\n=== Sample data ===")
    print("Columns:", list(train_dataset.columns))
    if len(train_dataset) > 0:
        sample = train_dataset.iloc[0]
        print("\nFirst sample:")
        print(f"  prompt: {str(sample['prompt'])[:200]}...")
        print(f"  data_source: {sample['data_source']}")
        print(f"  agent_name: {sample['agent_name']}")


if __name__ == "__main__":
    main()
