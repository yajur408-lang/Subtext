"""
Tweet Sentiment Analysis Pipeline
Comprehensive ML pipeline with feature engineering, embeddings, and model training
"""

import pandas as pd
import numpy as np
import re
import warnings
from scipy.sparse import hstack
warnings.filterwarnings('ignore')

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, accuracy_score, f1_score
)
import xgboost as xgb

# Sentiment analysis libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline as hf_pipeline

# Word2Vec
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for Word2Vec embeddings"""
    
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
    
    def fit(self, X, y=None):
        """Train Word2Vec model"""
        tokenized = [simple_preprocess(text, deacc=True) for text in X]
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=0
        )
        return self
    
    def transform(self, X):
        """Generate embeddings"""
        tokenized = [simple_preprocess(text, deacc=True) for text in X]
        embeddings = []
        for tokens in tokenized:
            if tokens:
                word_vectors = [
                    self.model.wv[word] 
                    for word in tokens 
                    if word in self.model.wv
                ]
                if word_vectors:
                    embeddings.append(np.mean(word_vectors, axis=0))
                else:
                    embeddings.append(np.zeros(self.vector_size))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)


class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract sentiment features using VADER and TextBlob"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract sentiment features"""
        features = []
        for text in X:
            # VADER scores
            vader_scores = self.vader.polarity_scores(str(text))
            # TextBlob scores
            blob = TextBlob(str(text))
            features.append([
                vader_scores['compound'],
                vader_scores['pos'],
                vader_scores['neu'],
                vader_scores['neg'],
                blob.sentiment.polarity,
                blob.sentiment.subjectivity
            ])
        return np.array(features)


class TweetSentimentAnalyzer:
    """Main class for tweet sentiment analysis"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.transformer_pipeline = None
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.scaler = StandardScaler()
        
    def extract_text_from_quotes(self, text):
        """Extract text inside quotation marks, or return cleaned text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Find text inside quotes
        quoted_text = re.findall(r'"([^"]*)"', text)
        if quoted_text:
            return ' '.join(quoted_text)
        return text
    
    def clean_text(self, text):
        """Remove special characters and clean text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags (keep the word)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower().strip()
    
    def get_vader_scores(self, text):
        """Get VADER sentiment scores"""
        scores = self.vader.polarity_scores(text)
        return {
            'vader_compound': scores['compound'],
            'vader_pos': scores['pos'],
            'vader_neu': scores['neu'],
            'vader_neg': scores['neg']
        }
    
    def get_textblob_scores(self, text):
        """Get TextBlob polarity and subjectivity"""
        blob = TextBlob(text)
        return {
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def get_transformer_scores(self, text):
        """Get transformer-based sentiment scores"""
        if self.transformer_pipeline is None:
            try:
                self.transformer_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
            except:
                # Fallback to a simpler model
                self.transformer_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    return_all_scores=True
                )
        
        try:
            results = self.transformer_pipeline(text[:512])  # Limit length
            scores = {item['label']: item['score'] for item in results[0]}
            return {
                'transformer_positive': scores.get('POSITIVE', scores.get('LABEL_2', 0)),
                'transformer_negative': scores.get('NEGATIVE', scores.get('LABEL_0', 0)),
                'transformer_neutral': scores.get('NEUTRAL', scores.get('LABEL_1', 0))
            }
        except:
            return {
                'transformer_positive': 0.0,
                'transformer_negative': 0.0,
                'transformer_neutral': 0.0
            }
    
    def categorize_sentiment(self, compound, polarity):
        """Convert numerical scores to categories"""
        if compound < -0.5 or polarity < -0.5:
            return "Negative"
        elif compound > 0.5 or polarity > 0.5:
            return "Positive"
        elif abs(compound) < 0.1 or abs(polarity) < 0.1:
            return "Neutral"
        elif (compound > 0.1 and compound < 0.5) or (polarity > 0.1 and polarity < 0.5):
            return "Playful"
        else:
            return "Sarcastic"
    
    def process_tweets(self, df, fit_transformers=False):
        """Process tweets and extract all features"""
        print("Processing tweets...")
        
        # Extract and clean text
        df['tweet_text'] = df.iloc[:, -1].apply(self.extract_text_from_quotes)
        df['cleaned_text'] = df['tweet_text'].apply(self.clean_text)
        
        # Get sentiment scores
        print("Extracting VADER scores...")
        vader_scores = df['cleaned_text'].apply(self.get_vader_scores)
        for key in ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg']:
            df[key] = vader_scores.apply(lambda x: x[key])
        
        print("Extracting TextBlob scores...")
        textblob_scores = df['cleaned_text'].apply(self.get_textblob_scores)
        df['textblob_polarity'] = textblob_scores.apply(lambda x: x['textblob_polarity'])
        df['textblob_subjectivity'] = textblob_scores.apply(lambda x: x['textblob_subjectivity'])
        
        print("Extracting Transformer scores...")
        if fit_transformers:
            transformer_scores = df['cleaned_text'].apply(self.get_transformer_scores)
            df['transformer_positive'] = transformer_scores.apply(lambda x: x['transformer_positive'])
            df['transformer_negative'] = transformer_scores.apply(lambda x: x['transformer_negative'])
            df['transformer_neutral'] = transformer_scores.apply(lambda x: x['transformer_neutral'])
        
        # Create sentiment categories
        df['sentiment_category'] = df.apply(
            lambda row: self.categorize_sentiment(
                row['vader_compound'], 
                row['textblob_polarity']
            ), axis=1
        )
        
        return df
    
    def create_tfidf_features(self, train_texts, test_texts=None):
        """Create TF-IDF features"""
        print("Creating TF-IDF features...")
        if test_texts is None:
            # Fit and transform on training data
            tfidf_features = self.tfidf_vectorizer.fit_transform(train_texts)
            return tfidf_features
        else:
            # Transform test data
            train_tfidf = self.tfidf_vectorizer.transform(train_texts)
            test_tfidf = self.tfidf_vectorizer.transform(test_texts)
            return train_tfidf, test_tfidf
    
    def tokenize_tweets(self, texts):
        """Tokenize tweets for Word2Vec"""
        return [simple_preprocess(text, deacc=True) for text in texts]
    
    def train_word2vec(self, train_texts):
        """Train Word2Vec model on training data only"""
        print("Training Word2Vec model...")
        tokenized = self.tokenize_tweets(train_texts)
        self.word2vec_model = Word2Vec(
            sentences=tokenized,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            sg=0  # CBOW
        )
        print(f"Word2Vec vocabulary size: {len(self.word2vec_model.wv)}")
    
    def get_word2vec_embeddings(self, texts):
        """Generate tweet embeddings using trained Word2Vec model"""
        tokenized = self.tokenize_tweets(texts)
        embeddings = []
        for tokens in tokenized:
            if tokens:
                word_vectors = [
                    self.word2vec_model.wv[word] 
                    for word in tokens 
                    if word in self.word2vec_model.wv
                ]
                if word_vectors:
                    embeddings.append(np.mean(word_vectors, axis=0))
                else:
                    embeddings.append(np.zeros(100))
            else:
                embeddings.append(np.zeros(100))
        return np.array(embeddings)
    
    def prepare_features(self, df, is_train=False):
        """Prepare all features for modeling"""
        # Sentiment features
        sentiment_features = [
            'vader_compound', 'vader_pos', 'vader_neu', 'vader_neg',
            'textblob_polarity', 'textblob_subjectivity'
        ]
        
        # Add transformer features if available
        if 'transformer_positive' in df.columns:
            sentiment_features.extend([
                'transformer_positive', 'transformer_negative', 'transformer_neutral'
            ])
        
        sentiment_df = df[sentiment_features].values
        
        # TF-IDF features
        if is_train:
            tfidf_features = self.create_tfidf_features(df['cleaned_text'].values)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(df['cleaned_text'].values)
        
        # Word2Vec embeddings
        w2v_embeddings = self.get_word2vec_embeddings(df['cleaned_text'].values)
        
        # Combine all features
        from scipy.sparse import hstack
        if isinstance(tfidf_features, np.ndarray):
            all_features = np.hstack([sentiment_df, tfidf_features.toarray(), w2v_embeddings])
        else:
            all_features = hstack([sentiment_df, tfidf_features, w2v_embeddings]).toarray()
        
        return all_features


def load_data():
    """Load training and validation data, including flirting datasets"""
    print("Loading data...")
    
    # Load main training data
    train_df = pd.read_csv('twitter_training.csv', header=None, 
                          names=['id', 'topic', 'sentiment', 'tweet'])
    val_df = pd.read_csv('twitter_validation.csv', header=None,
                        names=['id', 'topic', 'sentiment', 'tweet'])
    
    print(f"Main training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Load and process flirting_rated.csv
    try:
        print("\nLoading flirting_rated.csv...")
        flirting_rated = pd.read_csv('flirting_rated.csv')
        
        # Convert to match twitter_training format
        # Default mapping: polarity 0 = Neutral, 1 = Positive
        # If labels seem reversed, swap the mapping
        # You can adjust this based on your analysis
        polarity_mapping = {0: 'Neutral', 1: 'Positive'}  # Adjust if needed
        
        flirting_rated_processed = pd.DataFrame({
            'id': range(len(train_df), len(train_df) + len(flirting_rated)),
            'topic': 'Flirting',
            'sentiment': flirting_rated['polarity'].map(polarity_mapping),
            'tweet': flirting_rated['final_messages']
        })
        
        print(f"  Added {len(flirting_rated_processed)} samples from flirting_rated.csv")
        print(f"  Mapping: polarity 0→{polarity_mapping[0]}, 1→{polarity_mapping[1]}")
        train_df = pd.concat([train_df, flirting_rated_processed], ignore_index=True)
    except Exception as e:
        print(f"  Warning: Could not load flirting_rated.csv: {e}")
    
    # Load and process flirtation_dataset.csv
    try:
        print("\nLoading flirtation_dataset.csv...")
        flirtation_data = pd.read_csv('flirtation_dataset.csv')
        
        # Map scores to sentiment categories
        # Adjust these thresholds based on your analysis of the data
        # Current: score >= 50 = Positive (flirty), < 50 = Neutral
        # You can adjust these thresholds
        def map_score_to_sentiment(score):
            if score >= 50:  # Adjust threshold as needed
                return 'Positive'  # Flirty
            else:
                return 'Neutral'   # Not flirty
        
        flirtation_processed = pd.DataFrame({
            'id': range(len(train_df), len(train_df) + len(flirtation_data)),
            'topic': 'Flirting',
            'sentiment': flirtation_data['score'].apply(map_score_to_sentiment),
            'tweet': flirtation_data['message']
        })
        
        print(f"  Added {len(flirtation_processed)} samples from flirtation_dataset.csv")
        print(f"  Mapping: score >= 50 → Positive, < 50 → Neutral")
        train_df = pd.concat([train_df, flirtation_processed], ignore_index=True)
    except Exception as e:
        print(f"  Warning: Could not load flirtation_dataset.csv: {e}")
    
    print(f"\nTotal training samples after adding flirting data: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    return train_df, val_df


def visualize_sentiment_distribution(df, title="Sentiment Distribution"):
    """Visualize sentiment distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Target distribution
    sentiment_counts = df['sentiment'].value_counts()
    axes[0].bar(sentiment_counts.index, sentiment_counts.values)
    axes[0].set_title(f'{title} - Target Labels')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Feature distribution
    if 'vader_compound' in df.columns:
        axes[1].hist(df['vader_compound'], bins=50, alpha=0.7)
        axes[1].set_title(f'{title} - VADER Compound Score')
        axes[1].set_xlabel('Compound Score')
        axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(models, X_test, y_test, model_names):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model, name in zip(models, model_names):
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(models, X_test, y_test, model_names):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
    if len(models) == 1:
        axes = [axes]
    
    for model, name, ax in zip(models, model_names, axes):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_errors(model, X_test, y_test, test_df, model_name):
    """Analyze misclassifications"""
    y_pred = model.predict(X_test)
    misclassified = test_df[y_test != y_pred].copy()
    misclassified['predicted'] = y_pred[y_test != y_pred]
    misclassified['actual'] = y_test[y_test != y_pred]
    
    print(f"\n=== Error Analysis for {model_name} ===")
    print(f"Total misclassifications: {len(misclassified)}")
    print(f"Error rate: {len(misclassified)/len(test_df)*100:.2f}%")
    
    if len(misclassified) > 0:
        print("\nSample misclassified tweets:")
        for idx, row in misclassified.head(10).iterrows():
            print(f"\nTweet: {row['cleaned_text'][:100]}...")
            print(f"Actual: {row['actual']}, Predicted: {row['predicted']}")
    
    return misclassified


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def combine_features(tfidf_features, sentiment_features, w2v_features):
    """Combine different feature types"""
    # Convert sentiment and w2v to sparse if needed
    from scipy.sparse import csr_matrix
    
    # TF-IDF is already sparse
    # Convert sentiment and w2v to sparse matrices for efficient stacking
    sentiment_sparse = csr_matrix(sentiment_features)
    w2v_sparse = csr_matrix(w2v_features)
    
    # Combine all features
    combined = hstack([tfidf_features, sentiment_sparse, w2v_sparse]).toarray()
    return combined


def main():
    """Main execution function"""
    # Load data
    train_df, val_df = load_data()
    
    # Initialize analyzer for text preprocessing
    analyzer = TweetSentimentAnalyzer()
    
    # Process training data
    print("\n=== Processing Training Data ===")
    train_df = analyzer.process_tweets(train_df, fit_transformers=False)  # Skip transformer for speed
    
    # Process validation data
    print("\n=== Processing Validation Data ===")
    val_df = analyzer.process_tweets(val_df, fit_transformers=False)
    
    # Visualize training distribution
    visualize_sentiment_distribution(train_df, "Training Set")
    
    # Encode target variable - binary classification (Positive vs Others)
    train_df['target'] = (train_df['sentiment'] == 'Positive').astype(int)
    val_df['target'] = (val_df['sentiment'] == 'Positive').astype(int)
    
    # Prepare text data (cleaned_text column)
    X_train_text = train_df['cleaned_text'].values
    X_val_text = val_df['cleaned_text'].values
    y_train = train_df['target'].values
    y_val = val_df['target'].values
    
    # Create and fit feature extractors on training data only
    print("\n=== Creating and Fitting Feature Extractors ===")
    
    # TF-IDF
    print("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)
    
    # Sentiment features
    print("Extracting sentiment features...")
    sentiment_extractor = SentimentFeatureExtractor()
    X_train_sentiment = sentiment_extractor.fit_transform(X_train_text)
    X_val_sentiment = sentiment_extractor.transform(X_val_text)
    
    # Word2Vec
    print("Training Word2Vec...")
    w2v = Word2VecTransformer(vector_size=100)
    X_train_w2v = w2v.fit_transform(X_train_text)
    X_val_w2v = w2v.transform(X_val_text)
    
    # Combine all features
    print("Combining features...")
    X_train_processed = combine_features(X_train_tfidf, X_train_sentiment, X_train_w2v)
    X_val_processed = combine_features(X_val_tfidf, X_val_sentiment, X_val_w2v)
    
    print(f"Training features shape: {X_train_processed.shape}")
    print(f"Validation features shape: {X_val_processed.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_val_scaled = scaler.transform(X_val_processed)
    
    # Create model pipelines
    print("\n=== Creating Model Pipelines ===")
    models = {}
    model_names = []
    
    # Logistic Regression Pipeline
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
    ])
    models['Logistic Regression'] = lr_pipeline
    model_names.append('Logistic Regression')
    
    # Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    models['Random Forest'] = rf_pipeline
    model_names.append('Random Forest')
    
    # XGBoost Pipeline
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'))
    ])
    models['XGBoost'] = xgb_pipeline
    model_names.append('XGBoost')
    
    # Train models
    print("\n=== Training Models ===")
    trained_models = {}
    
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train_scaled, y_train)
        trained_models[name] = pipeline
    
    # Evaluate models
    print("\n=== Model Evaluation ===")
    results = []
    
    for name, model in trained_models.items():
        # Test set evaluation
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else 0
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'AUC': auc
        })
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(classification_report(y_val, y_pred))
        
        # Cross-validation with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=tscv, scoring='f1', n_jobs=-1)
        print(f"  CV F1 Score (mean): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Visualizations
    print("\n=== Creating Visualizations ===")
    plot_roc_curves(list(trained_models.values()), X_val_scaled, y_val, model_names)
    plot_confusion_matrices(list(trained_models.values()), X_val_scaled, y_val, model_names)
    
    # Error analysis
    print("\n=== Error Analysis ===")
    for name, model in trained_models.items():
        analyze_errors(model, X_val_scaled, y_val, val_df, name)
    
    # Feature importance (for tree-based models)
    print("\n=== Feature Importance ===")
    feature_names = (
        [f'tfidf_{i}' for i in range(1000)] +
        ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg',
         'textblob_polarity', 'textblob_subjectivity'] +
        [f'w2v_{i}' for i in range(100)]
    )
    
    for name, model in trained_models.items():
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            plot_feature_importance(
                model.named_steps['classifier'], 
                feature_names, 
                name
            )
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_results.csv', index=False)
    print("\nResults saved to model_results.csv")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()

