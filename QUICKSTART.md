# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Setup NLTK Data (One-time)

```bash
python setup_nltk.py
```

## Step 3: Run the Analysis

```bash
python tweet_sentiment_analysis.py
```

This will:
- Process ~74K training tweets and ~1.7K validation tweets
- Extract features (VADER, TextBlob, TF-IDF, Word2Vec)
- Train 3 models (Logistic Regression, Random Forest, XGBoost)
- Generate visualizations and save results

**Expected runtime**: 10-30 minutes depending on your hardware

## Step 4: View Results Interactively

```bash
streamlit run viewer.py
```

Open your browser to the URL shown (usually http://localhost:8501)

## Output Files

After running the analysis, you'll find:
- `model_results.csv` - Performance metrics for all models
- `roc_curves.png` - ROC curve comparison
- `confusion_matrices.png` - Confusion matrices
- `feature_importance_*.png` - Feature importance plots
- `training_set.png` - Data distribution

## Troubleshooting

**Out of memory?**
- Reduce `max_features=1000` to `max_features=500` in the TF-IDF vectorizer

**Too slow?**
- The transformer-based sentiment is disabled by default (`fit_transformers=False`)
- If still slow, reduce the dataset size for testing

**NLTK errors?**
- Run `python setup_nltk.py` again
- Or manually: `python -m nltk.downloader punkt brown wordnet`

