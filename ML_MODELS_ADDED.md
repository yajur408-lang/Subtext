# ML Models Added to Sentiment Analyzer

## What's New

The Sentiment Analyzer now includes **three ML models** in addition to the transformer models:

1. **Logistic Regression** - Fast, interpretable linear model
2. **Random Forest** - Ensemble tree-based model
3. **XGBoost** - Gradient boosting model

## How It Works

### Model Training
- On first use, the ML models are trained on your training data (`twitter_training.csv`)
- Models are saved to `ml_models.pkl` for faster subsequent loads
- Training uses the same feature extraction pipeline as the main analysis:
  - TF-IDF vectorization
  - VADER + TextBlob sentiment features
  - Word2Vec embeddings

### Prediction
- User input is processed through the same feature extraction pipeline
- All three ML models make predictions
- Results are combined with transformer model predictions
- Consensus sentiment is calculated across all models

## Usage

1. Go to the **Sentiment Analyzer** tab
2. Enter your text
3. Select which models to use (checkboxes in sidebar):
   - **Transformer Models**: Twitter-RoBERTa, FinBERT, BERT
   - **ML Models**: Logistic Regression, Random Forest, XGBoost
4. Click "Analyze Sentiment"
5. View results from all selected models

## Model Comparison

| Model Type | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| Transformer Models | Slower | High | Best for nuanced sentiment |
| Logistic Regression | Fast | Good | Quick predictions |
| Random Forest | Medium | Very Good | Robust predictions |
| XGBoost | Medium | Excellent | Best overall performance |

## Features

- **Consensus Sentiment**: Shows agreement across all models
- **Confidence Scores**: Each model provides confidence level
- **Detailed Scores**: View probability distributions
- **Keyword Detection**: Enhanced with keyword matching
- **Sentiment Categories**: Maps to Sarcastic, Playful, Funny, Flirty, Angry, Neutral

## Requirements

- **scikit-learn**: `pip install scikit-learn`
- **xgboost**: `pip install xgboost`
- **Training data**: `twitter_training.csv` (for initial training)

## Performance

- **First run**: ~2-5 minutes (trains models)
- **Subsequent runs**: Instant (uses saved models)
- **Model file**: `ml_models.pkl` (~50-100 MB)

## Notes

- ML models are trained on binary classification (Positive vs Others)
- Results are mapped to extended sentiment categories
- Models are cached using Streamlit's `@st.cache_resource`
- If training data changes, delete `ml_models.pkl` to retrain

