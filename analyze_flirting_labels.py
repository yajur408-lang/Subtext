"""
Analyze flirting dataset labels to understand the correct mapping
"""

import pandas as pd

print("=" * 70)
print("Analyzing Flirting Dataset Labels")
print("=" * 70)

# Analyze flirting_rated.csv
print("\n1. FLIRTING_RATED.CSV")
print("-" * 70)
try:
    df1 = pd.read_csv('flirting_rated.csv')
    print(f"Total samples: {len(df1)}")
    print(f"\nPolarity distribution:")
    print(df1['polarity'].value_counts().sort_index())
    
    print(f"\nSample messages with polarity=0 (NOT flirty):")
    for msg in df1[df1['polarity'] == 0]['final_messages'].head(5):
        print(f"  - {msg[:80]}")
    
    print(f"\nSample messages with polarity=1 (FLIRTY):")
    for msg in df1[df1['polarity'] == 1]['final_messages'].head(5):
        print(f"  - {msg[:80]}")
except Exception as e:
    print(f"Error: {e}")

# Analyze flirtation_dataset.csv
print("\n\n2. FLIRTATION_DATASET.CSV")
print("-" * 70)
try:
    df2 = pd.read_csv('flirtation_dataset.csv')
    print(f"Total samples: {len(df2)}")
    print(f"\nScore distribution:")
    print(df2['score'].describe())
    print(f"\nScore ranges:")
    print(f"  0-30: {len(df2[df2['score'] <= 30])} samples")
    print(f"  31-50: {len(df2[(df2['score'] > 30) & (df2['score'] <= 50)])} samples")
    print(f"  51-70: {len(df2[(df2['score'] > 50) & (df2['score'] <= 70)])} samples")
    print(f"  71-100: {len(df2[df2['score'] > 70])} samples")
    
    print(f"\nSample messages with score 0-30 (LOW flirtation):")
    for msg in df2[df2['score'] <= 30]['message'].head(3):
        print(f"  - {msg[:80]}")
    
    print(f"\nSample messages with score 31-50 (MODERATE flirtation):")
    for msg in df2[(df2['score'] > 30) & (df2['score'] <= 50)]['message'].head(3):
        print(f"  - {msg[:80]}")
    
    print(f"\nSample messages with score 51-70 (HIGH flirtation):")
    for msg in df2[(df2['score'] > 50) & (df2['score'] <= 70)]['message'].head(3):
        print(f"  - {msg[:80]}")
    
    print(f"\nSample messages with score 71-100 (VERY HIGH flirtation):")
    for msg in df2[df2['score'] > 70]['message'].head(3):
        print(f"  - {msg[:80]}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 70)
print("RECOMMENDED MAPPING:")
print("=" * 70)
print("""
For flirting_rated.csv:
  - polarity 0 = Neutral (not flirty)
  - polarity 1 = Positive/Flirty (flirty)

For flirtation_dataset.csv:
  - score 0-30 = Neutral (not flirty)
  - score 31-60 = Neutral (moderate, not clearly flirty)
  - score 61-80 = Positive (flirty)
  - score 81-100 = Positive (very flirty)

OR use a more nuanced approach:
  - score 0-40 = Neutral
  - score 41-70 = Positive (moderately flirty)
  - score 71-100 = Positive (very flirty)
""")

