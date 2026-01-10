"""
Interactive script to analyze and fix label mappings for flirting datasets
"""

import pandas as pd

print("=" * 70)
print("FLIRTING DATASET LABEL ANALYZER")
print("=" * 70)

# Analyze flirting_rated.csv
print("\n1. ANALYZING flirting_rated.csv")
print("-" * 70)
try:
    df1 = pd.read_csv('flirting_rated.csv')
    print(f"Total samples: {len(df1)}")
    print(f"\nPolarity distribution:")
    print(df1['polarity'].value_counts().sort_index())
    
    print(f"\n--- Sample messages with polarity=0 ---")
    samples_0 = df1[df1['polarity'] == 0]['final_messages'].head(10).tolist()
    for i, msg in enumerate(samples_0, 1):
        print(f"{i}. {msg}")
    
    print(f"\n--- Sample messages with polarity=1 ---")
    samples_1 = df1[df1['polarity'] == 1]['final_messages'].head(10).tolist()
    for i, msg in enumerate(samples_1, 1):
        print(f"{i}. {msg}")
    
    print("\n" + "-" * 70)
    print("QUESTION: Looking at the samples above:")
    print("  - Are polarity=0 messages actually NOT flirty?")
    print("  - Are polarity=1 messages actually FLIRTY?")
    print("\nIf NO, the labels are reversed!")
    print("  → Change mapping to: {0: 'Positive', 1: 'Neutral'}")
    print("If YES, current mapping is correct: {0: 'Neutral', 1: 'Positive'}")
    
except Exception as e:
    print(f"Error loading flirting_rated.csv: {e}")

# Analyze flirtation_dataset.csv
print("\n\n2. ANALYZING flirtation_dataset.csv")
print("-" * 70)
try:
    df2 = pd.read_csv('flirtation_dataset.csv')
    print(f"Total samples: {len(df2)}")
    print(f"\nScore statistics:")
    print(df2['score'].describe())
    
    # Show samples by score ranges
    ranges = [
        (0, 30, "LOW (0-30)"),
        (31, 50, "MODERATE (31-50)"),
        (51, 70, "HIGH (51-70)"),
        (71, 100, "VERY HIGH (71-100)")
    ]
    
    for min_score, max_score, label in ranges:
        subset = df2[(df2['score'] >= min_score) & (df2['score'] <= max_score)]
        print(f"\n--- {label} flirtation ({len(subset)} samples) ---")
        for msg in subset['message'].head(5):
            print(f"  • {msg}")
    
    print("\n" + "-" * 70)
    print("QUESTION: What score threshold should separate 'Neutral' from 'Positive'?")
    print("\nCurrent threshold: score >= 50 → Positive")
    print("\nTry different thresholds:")
    for threshold in [40, 50, 60, 70]:
        pos_count = len(df2[df2['score'] >= threshold])
        neu_count = len(df2[df2['score'] < threshold])
        print(f"  Threshold {threshold}: {pos_count} Positive, {neu_count} Neutral")
    
except Exception as e:
    print(f"Error loading flirtation_dataset.csv: {e}")

print("\n" + "=" * 70)
print("TO FIX THE MAPPING:")
print("=" * 70)
print("""
1. Edit tweet_sentiment_analysis.py:
   - Line ~490: Change polarity_mapping if needed
   - Line ~510: Adjust score threshold if needed

2. Edit viewer.py:
   - Line ~656: Change polarity_mapping if needed  
   - Line ~666: Adjust score threshold if needed

3. After fixing, delete ml_models.pkl to retrain with correct labels
4. Run: python tweet_sentiment_analysis.py
""")

