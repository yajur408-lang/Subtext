"""
Configurable label mapping for flirting datasets
Adjust these mappings based on your analysis of the data
"""

# Mapping for flirting_rated.csv
# polarity: 0 = not flirty, 1 = flirty
FLIRTING_RATED_MAPPING = {
    0: 'Neutral',  # Not flirty
    1: 'Positive'  # Flirty
}

# Alternative mapping if labels are reversed:
# FLIRTING_RATED_MAPPING = {
#     0: 'Positive',  # Actually flirty
#     1: 'Neutral'   # Actually not flirty
# }

# Mapping for flirtation_dataset.csv based on score thresholds
# Adjust these thresholds based on your analysis
FLIRTATION_SCORE_THRESHOLDS = {
    'Neutral': (0, 40),      # Low scores = not flirty
    'Positive': (41, 100)    # High scores = flirty
}

# Alternative thresholds (more nuanced):
# FLIRTATION_SCORE_THRESHOLDS = {
#     'Neutral': (0, 30),      # Very low = not flirty
#     'Neutral': (31, 50),     # Low-moderate = neutral
#     'Positive': (51, 70),    # Moderate-high = flirty
#     'Positive': (71, 100)    # Very high = very flirty
# }

def map_flirting_rated_polarity(polarity):
    """Map polarity from flirting_rated.csv to sentiment"""
    return FLIRTING_RATED_MAPPING.get(polarity, 'Neutral')

def map_flirtation_score(score):
    """Map score from flirtation_dataset.csv to sentiment"""
    for sentiment, (min_score, max_score) in FLIRTATION_SCORE_THRESHOLDS.items():
        if min_score <= score <= max_score:
            return sentiment
    return 'Neutral'  # Default

