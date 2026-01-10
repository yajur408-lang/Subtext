# Fixing Label Mapping for Flirting Datasets

## Current Mapping (May Be Inaccurate)

### flirting_rated.csv
- `polarity = 0` → `Neutral` (not flirty)
- `polarity = 1` → `Positive` (flirty)

### flirtation_dataset.csv
- `score >= 50` → `Positive` (flirty)
- `score < 50` → `Neutral` (not flirty)

## How to Fix

### Option 1: Analyze the Data First

Run this to see sample messages and understand the labels:
```bash
python analyze_flirting_labels.py
```

This will show you:
- Distribution of labels
- Sample messages for each label
- Help you determine if the mapping is correct

### Option 2: Adjust Mapping in Code

Edit `tweet_sentiment_analysis.py` and `viewer.py`:

**For flirting_rated.csv**, change the mapping:
```python
# If labels are reversed, swap them:
polarity_mapping = {0: 'Positive', 1: 'Neutral'}  # Reversed
```

**For flirtation_dataset.csv**, adjust the threshold:
```python
def map_score_to_sentiment(score):
    if score >= 60:  # Change threshold (try 40, 50, 60, 70)
        return 'Positive'
    else:
        return 'Neutral'
```

### Option 3: Use More Nuanced Mapping

Instead of binary (Positive/Neutral), you could use:
- `score 0-30` → `Neutral`
- `score 31-60` → `Neutral` (or create a new category)
- `score 61-80` → `Positive`
- `score 81-100` → `Positive`

### Option 4: Manual Review

1. Open the CSV files
2. Review sample messages for each label/score
3. Determine what the labels actually mean
4. Update the mapping accordingly

## Quick Fix Locations

### In `tweet_sentiment_analysis.py` (line ~490):
```python
polarity_mapping = {0: 'Neutral', 1: 'Positive'}  # Adjust here
```

### In `tweet_sentiment_analysis.py` (line ~510):
```python
if score >= 50:  # Adjust threshold here
    return 'Positive'
```

### In `viewer.py` (line ~656):
```python
'sentiment': flirting_rated['polarity'].map({0: 'Neutral', 1: 'Positive'})
```

### In `viewer.py` (line ~666):
```python
if score >= 70:  # Adjust threshold here
    return 'Positive'
```

## Testing Your Mapping

After changing the mapping:
1. Run: `python analyze_flirting_labels.py` to verify
2. Check a few sample messages manually
3. Run the training: `python tweet_sentiment_analysis.py`
4. Review the results to see if labels make sense

## Common Issues

1. **Labels Reversed**: If polarity 0 seems flirty and 1 seems not flirty, swap the mapping
2. **Wrong Threshold**: If too many/too few messages are marked as Positive, adjust the score threshold
3. **Need More Categories**: Consider adding "Flirty" as a separate category instead of just "Positive"

