# Active Sampling for LLM Leaderboards

This repository implements an **Active Sampling** strategy for ranking Large Language Models (LLMs). Instead of random matchmaking, which wastes budget on "solved" comparisons (e.g., Rank 1 vs Rank 50), this system intelligently selects the next pair of models to evaluate.

The goal is to **maximize information gain** per vote, ensuring the leaderboard converges to accurate rankings as fast as possible.


---

## Usage

### 1. Requirements
```bash
pip install pandas
```

### 2. Have Leaderboard ready as given CSV

### 3. Run the python script
```bash
python code.py
```

---

### The code outputs a list of matchups at a time , since we may need the priorities in advance, in case the scores are calculated in batches and not instantaneously. 
