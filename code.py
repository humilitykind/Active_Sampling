import random

def get_next_match_smart(models, epsilon=0.20):
    """
    epsilon: Probability of running "Exploration" (Student-Teacher mode).
             Otherwise runs "Exploitation" (Cluster-Buster).
    """
    
    # --- STRATEGY A: Student-Teacher (Inverse Vote Probability) ---
    # We use this instead of a hard " < 50 votes" rule.
    # Models with fewer votes get picked much more often.
    if random.random() < epsilon:
        # 1. Calculate Weights: 1 / (votes + 1)**2 to avoid divide-by-zero
        weights = [1 / (m.votes + 1)**2 for m in models]
        
        # 2. Pick Model A (The Student) based on weights
        # k=1 returns a list, so we take [0]
        student = random.choices(models, weights=weights, k=1)[0]
        
        # 3. Pick Model B (The Teacher/Anchor)
        # Find the most stable model (Smallest CI Range) that isn't the student
        potential_anchors = [m for m in models if m.name != student.name]
        
        # Sort by stability (Upper - Lower)
        # We pick from top 3 to keep it slightly varied
        best_anchors = sorted(potential_anchors, key=lambda m: (m.upper - m.lower))[:3]
        teacher = random.choice(best_anchors)
        
        return student.name, teacher.name, f"Student-Teacher (Student: {student.votes} votes)"

    # --- STRATEGY B: Cluster-Buster (Variance Reduction) ---
    # (Standard logic: Target overlaps)
    candidates = []
    total_weight = 0
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1 = models[i]
            m2 = models[j]
            overlap = min(m1.upper, m2.upper) - max(m1.lower, m2.lower)
            
            if overlap > 0:
                weight = overlap ** 2
                candidates.append((m1, m2, weight))
                total_weight += weight
    
    if total_weight > 0:
        pick = random.uniform(0, total_weight)
        current = 0
        for m1, m2, weight in candidates:
            current += weight
            if current >= pick:
                return m1.name, m2.name, f"Cluster-Buster (Overlap Mass: {weight:.0f})"

    # Fallback
    m1, m2 = random.sample(models, 2)
    return m1.name, m2.name, "Random"


import pandas as pd
import random

# file_path = "/Users/arshitmankodi/Documents/AI4Bharat/Active_Sampling/Leaderboard_models - Sheet1.csv"

# --- 1. Define the Model Class ---
class Model:
    def __init__(self, name, score, ci_upper, ci_lower, votes):
        self.name = name
        self.score = score
        self.votes = votes
        
        # Use exact bounds from the parsed CSV
        # ci_lower is negative (e.g., -39), so we add it to the score
        self.upper = score + ci_upper
        self.lower = score + ci_lower 

    def __repr__(self):
        return f"{self.name} (Range: [{self.lower:.0f}, {self.upper:.0f}])"

# --- 2. Pandas Loader & Parser ---
def load_models_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    models = []
    
    for _, row in df.iterrows():
        # Parse "47 / -47" string
        try:
            ci_str = str(row['CI'])
            if '/' in ci_str:
                parts = ci_str.split('/')
                c_up = float(parts[0].strip())
                c_low = float(parts[1].strip()) # Usually negative, e.g. -47.0
            else:
                # Fallback for simple numbers
                val = float(ci_str)
                c_up = val
                c_low = -val
        except ValueError:
            print(f"Skipping row with bad CI format: {row['CI']}")
            continue

        models.append(Model(
            name=row['Model'],
            score=float(row['Score']),
            ci_upper=c_up,
            ci_lower=c_low,
            votes=int(row['votes'])
        ))
        
    return models




# --- 4. Execution ---
if __name__ == "__main__":
    file_path = '/Users/arshitmankodi/Documents/AI4Bharat/Active_Sampling/Leaderboard_models - Sheet1.csv'
    
    # Load and Parse
    models = load_models_from_csv(file_path)
    print(f"Successfully loaded {len(models)} models.\n")
    
    # Run Simulation
    print(f"{'Model A':<30} vs {'Model B':<30} | {'Reason'}")
    print("-" * 100)
    
    for _ in range(5):
        m1, m2, reason = get_next_match_smart(models)
        print(f"{m1:<30} vs {m2:<30} | {reason}")