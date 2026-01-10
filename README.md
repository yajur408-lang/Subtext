# Tweet Sentiment Analysis Pipeline

A comprehensive machine learning pipeline for tweet sentiment analysis with feature engineering, multiple embedding techniques, and model evaluation.

## Features

- **Feature Engineering**:
  - Text cleaning (removes special characters, URLs, mentions)
  - Extracts text from quotation marks
  - VADER sentiment analysis (compound, positive, neutral, negative scores)
  - TextBlob sentiment analysis (polarity and subjectivity)
  - Transformer-based sentiment scores (optional)
  - Sentiment categorization (Positive, Negative, Neutral, Playful, Sarcastic)

- **Embeddings**:
  - TF-IDF vectorization (fitted only on training data)
  - Word2Vec embeddings (trained only on training data)

- **Models**:
  - Logistic Regression
  - Random Forest
  - XGBoost

- **Evaluation**:
  - Cross-validation with TimeSeriesSplit
  - ROC curves
  - Confusion matrices
  - Error analysis
  - Feature importance visualization

- **Interactive Viewer**:
  - Streamlit-based interactive viewer
  - Filter by sentiment, target, topic
  - Search functionality
  - Color-coded labels

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (for TextBlob):
```python
import nltk
nltk.download('punkt')
nltk.download('brown')
```

## Usage

### 1. Run the Main Analysis Pipeline

```bash
python tweet_sentiment_analysis.py
```

This will:
- Load and preprocess the training and validation data
- Extract features using VADER, TextBlob, and Transformers
- Create TF-IDF and Word2Vec embeddings
- Train Logistic Regression, Random Forest, and XGBoost models
- Evaluate models with cross-validation
- Generate visualizations (ROC curves, confusion matrices)
- Perform error analysis
- Save results to `model_results.csv`

### 2. Run the Interactive Viewer

```bash
streamlit run viewer.py
```

This will open a web interface where you can:
- Filter tweets by sentiment, target, and topic
- Search within tweets
- View color-coded sentiment labels
- Verify sample sizes (50+ samples per class)
- Download filtered data

## Data Format

The pipeline expects CSV files with the following format:
- `twitter_training.csv`: Training data
- `twitter_validation.csv`: Validation data

Each CSV should have columns (no header):
- Column 1: ID
- Column 2: Topic
- Column 3: Sentiment (Positive, Negative, Neutral, Irrelevant)
- Column 4: Tweet text

## Output Files

- `model_results.csv`: Model performance metrics
- `roc_curves.png`: ROC curve comparison
- `confusion_matrices.png`: Confusion matrices for all models
- `feature_importance_*.png`: Feature importance plots (for tree-based models)
- `training_set.png`: Training data distribution visualization

## Important Notes

1. **Data Leakage Prevention**:
   - TF-IDF is fitted only on training data
   - Word2Vec is trained only on training data
   - Test data is transformed separately
   - TimeSeriesSplit is used for cross-validation

2. **Performance**:
   - Transformer-based sentiment analysis is optional and can be slow
   - Set `fit_transformers=False` in `process_tweets()` to skip transformer features
   - Word2Vec training can take time for large datasets

3. **Memory**:
   - Large datasets may require significant memory
   - Consider reducing `max_features` in TF-IDF if memory is limited

## Model Pipeline Structure

The pipeline follows sklearn best practices:
- Preprocessing steps are wrapped in transformers
- Models use Pipeline for consistency
- ColumnTransformer is used for combining different feature types
- All preprocessing is fitted on training data only

## Troubleshooting

1. **Import errors**: Make sure all dependencies are installed
2. **Memory errors**: Reduce `max_features` in TF-IDF or use smaller datasets
3. **Slow execution**: Disable transformer features or reduce dataset size
4. **NLTK errors**: Run `nltk.download('punkt')` and `nltk.download('brown')`

## License

This project is for educational purposes.

