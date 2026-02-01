import pandas as pd
import random

# --- Configuration ---
CSV_PATH = 'Leaderboard_models - Sheet1.csv'
EPSILON = 0.20  # 20% Exploration (Student-Teacher), 80% Exploitation (Cluster-Buster)
ALPHA = 2.0     # Power law strength. Higher = Stronger bias towards low-vote models.

class Model:
    def __init__(self, name, score, ci_upper, ci_lower, votes):
        self.name = name
        self.score = score
        self.votes = votes
        # Calculate absolute bounds based on score +/- CI
        self.upper = score + ci_upper
        self.lower = score + ci_lower 

    def __repr__(self):
        return f"{self.name} (Votes: {self.votes}, CI Width: {self.upper-self.lower:.1f})"

def load_models_from_csv(file_path):
    df = pd.read_csv(file_path)
    models = []
    
    for _, row in df.iterrows():
        try:
            # Parse asymmetric CI formats like "47 / -47"
            ci_str = str(row['CI'])
            if '/' in ci_str:
                parts = ci_str.split('/')
                c_up = float(parts[0].strip())
                c_low = float(parts[1].strip()) # Usually negative
            else:
                val = float(ci_str)
                c_up = val
                c_low = -val
        except ValueError:
            continue

        models.append(Model(
            name=row['Model'],
            score=float(row['Score']),
            ci_upper=c_up,
            ci_lower=c_low,
            votes=int(row['votes'])
        ))
    return models

def get_next_match_smart(models, epsilon=EPSILON, alpha=ALPHA):
    """
    Hybrid Active Sampler:
    1. Student-Teacher: Power-law exploration for low-vote models.
    2. Cluster-Buster: Variance reduction for ambiguous tiers.
    """
    
    # --- STRATEGY A: Student-Teacher (Exploration) ---
    if random.random() < epsilon:
        # 1. Power Law Weighting: Weight = 1 / (votes + 1)^alpha
        # This heavily penalizes high vote counts.
        weights = [1 / ((m.votes + 1) ** alpha) for m in models]
        
        # 2. Pick The Student (New/Low-Data Model)
        student = random.choices(models, weights=weights, k=1)[0]
        
        # 3. Pick The Teacher (The Anchor)
        # Find potential opponents (excluding the student themselves)
        potential_anchors = [m for m in models if m.name != student.name]
        
        # Sort opponents by Stability (CI Width: Upper - Lower)
        # We pick from the top 3 most stable models to act as the "Ruler"
        best_anchors = sorted(potential_anchors, key=lambda m: (m.upper - m.lower))[:3]
        teacher = random.choice(best_anchors)
        
        return student.name, teacher.name, f"Student-Teacher (Student: {student.votes} votes, alpha={alpha})"

    # --- STRATEGY B: Cluster-Buster (Exploitation) ---
    candidates = []
    total_weight = 0
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1 = models[i]
            m2 = models[j]
            
            # Overlap Calculation: Min(Tops) - Max(Bottoms)
            overlap = min(m1.upper, m2.upper) - max(m1.lower, m2.lower)
            
            if overlap > 0:
                # Square the overlap to prioritize high-ambiguity pairs
                weight = overlap ** 2
                candidates.append((m1, m2, weight))
                total_weight += weight
    
    # Weighted Random Selection from Candidates
    if total_weight > 0:
        pick = random.uniform(0, total_weight)
        current = 0
        for m1, m2, weight in candidates:
            current += weight
            if current >= pick:
                return m1.name, m2.name, f"Cluster-Buster (Ambiguity Mass: {weight**0.5:.0f})"

    # Fallback: Random Sampling (if leaderboard is perfectly resolved)
    m1, m2 = random.sample(models, 2)
    return m1.name, m2.name, "Random (Leaderboard Resolved)"

if __name__ == "__main__":
    models = load_models_from_csv(CSV_PATH)
    print(f"Loaded {len(models)} models.\n")
    
    print(f"{'Model A':<30} vs {'Model B':<30} | {'Strategy'}")
    print("-" * 100)
    
    # Simulate 10 Matches
    for _ in range(10):
        m1, m2, reason = get_next_match_smart(models)
        print(f"{m1:<30} vs {m2:<30} | {reason}")