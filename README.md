# Active Sampling for LLM Leaderboards

This repository implements an **Active Sampling** strategy for ranking Large Language Models (LLMs). Instead of random matchmaking, which wastes budget on "solved" comparisons (e.g., Rank 1 vs Rank 50), this system intelligently selects the next pair of models to evaluate.

The goal is to **maximize information gain** per vote, ensuring the leaderboard converges to accurate rankings as fast as possible.

---

##  The Evolution of the Algorithm

Our sampling strategy evolved in three distinct phases to address specific statistical challenges found in real-world leaderboards.

### Phase 1: The "Cluster-Buster" (Variance Reduction)
**The Problem:** In a typical leaderboard, models often form "clusters" where ranks are indistinguishable (e.g., Model A is 1209 ± 30, Model B is 1198 ± 40). Random sampling wastes votes on models that are already clearly separated (e.g., Rank 1 vs Rank 10), while the "messy middle" remains unresolved.

**The Solution:** We calculate the **Overlap** of 95% Confidence Intervals (CI) for every pair.
* **Formula:** `Overlap = Min(Upper_A, Upper_B) - Max(Lower_A, Lower_B)`
* **Logic:** If `Overlap > 0`, the pair is ambiguous. We assign sampling probability proportional to `Overlap^2`.
* **Result:** The system aggressively "spams" matches within tied clusters until they separate, ignoring resolved pairs.

### Phase 2: Solving "Starvation" (Greedy Epsilon)
**The Problem:** The "Cluster-Buster" relies on existing variance data. A brand new model with 0 votes has undefined variance and might be ignored, or conversely, a model might get "lucky" early on and land in a gap where it stops receiving votes ("Starvation").

**The Solution:** We introduced an **Epsilon-Greedy** mechanism.
* **Logic:** `X%` of the time (e.g., 20%), ignore the complex math and just pick the model with the **fewest votes**.
* **Result:** Ensures every model gets a minimum baseline of attention, preventing infinite starvation.

### Phase 3: The "Student-Teacher" & Power Laws (Final Approach)
**The Problem:** Simply picking the lowest-vote model and pairing it randomly is inefficient. If a noisy new model fights another noisy new model, the result is low-quality data ("The Blind Leading the Blind").

**The Solution:** We refined the exploration phase with two upgrades:
1.  **Power-Law Selection (The "Student"):** Instead of a hard threshold, we pick candidates using **Inverse Probability Weighting** raised to a power (`alpha`).
    * `Weight = 1 / (votes + 1)^alpha`
    * With `alpha=3`, a model with 10 votes is ~1000x more likely to be picked than one with 100 votes.
2.  **Anchor Opponents (The "Teacher"):** When a new model ("Student") is picked, we force it to fight an **"Anchor"**—a model with the smallest CI (lowest variance).
    * **Result:** The "Student" is measured against a stable "Ruler," maximizing the signal regarding their true skill level.

---

## 🛠️ The Final Algorithm

The script implements a hybrid strategy that runs on every request:

1.  **Exploration Mode (Epsilon % chance):**
    * **Selection:** Pick a "Student" using **Power Law Weighting** (heavily favoring low-vote models).
    * **Opponent:** Pick a "Teacher" (Anchor) by finding models with the tightest Confidence Intervals.
    * *Goal: Rapidly place new models.*

2.  **Exploitation Mode (1 - Epsilon % chance):**
    * **Selection:** Calculate CI Overlaps for all pairs.
    * **Weighting:** Probability $\propto$ `Overlap^2`.
    * *Goal: Break ties and resolve ordering among established models.*

---

## 🚀 Usage

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
