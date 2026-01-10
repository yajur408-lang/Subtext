# Sentiment Analyzer Guide

## New Features

The viewer now includes a **Sentiment Analyzer** tab that allows you to input custom text and analyze it using multiple transformer models.

## Available Models

1. **Twitter-RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
   - Optimized for Twitter/social media text
   - Labels: Negative, Neutral, Positive

2. **FinBERT** (`ProsusAI/finbert`)
   - Financial sentiment analysis model
   - Labels: Positive, Negative, Neutral

3. **BERT Multilingual** (`nlptown/bert-base-multilingual-uncased-sentiment`)
   - Multilingual sentiment analysis
   - Labels: 1-5 star ratings

## Sentiment Categories

The analyzer maps model outputs to these extended categories:

- **Sarcastic** 🔴 - Sarcastic or ironic content
- **Playful** 🔵 - Light-hearted, fun content
- **Funny** 🟡 - Humorous, comedic content
- **Flirty** 🟣 - Flirtatious or romantic content
- **Angry** 🔴 - Angry, frustrated, or negative content
- **Sadness** 🟣 - Sad, depressed, or melancholic content
- **Dark Humour** ⚫ - Dark, morbid, or twisted humor about serious topics
- **Neutral** ⚪ - Neutral or balanced content

## How It Works

1. **Text Input**: Enter or paste text in the text area
2. **Model Selection**: Choose which models to use (all enabled by default)
3. **Analysis**: Click "Analyze Sentiment" to process the text
4. **Results**: View sentiment predictions from each model with confidence scores
5. **Consensus**: See the overall consensus sentiment across all models

## Features

- **Multi-Model Analysis**: Compare results from 3 different transformer models
- **Keyword Detection**: Enhanced detection using keyword matching
- **Confidence Scores**: See how confident each model is in its prediction
- **Detailed Scores**: View raw model outputs and probability distributions
- **Consensus Sentiment**: Get an aggregated sentiment across all models
- **Visual Display**: Color-coded sentiment cards for easy visualization

## Usage Tips

1. **First Run**: Models will download automatically on first use (may take a few minutes)
2. **GPU Support**: If you have a GPU, models will use it automatically for faster inference
3. **Text Length**: Very long texts (>512 tokens) will be truncated
4. **Model Selection**: You can enable/disable specific models in the sidebar
5. **Keyword Override**: If keywords are detected, they may override model predictions

## Example Use Cases

- Analyze social media posts
- Check sentiment of customer reviews
- Evaluate tweet sentiment
- Analyze product feedback
- Check email tone

## Technical Details

- Models are cached using `@st.cache_resource` for faster subsequent runs
- GPU acceleration is automatically used if available
- Text is truncated to 512 tokens for model compatibility
- Keyword detection runs before model mapping for better accuracy

## Troubleshooting

**Models not loading?**
- Check your internet connection (models download from Hugging Face)
- Ensure you have enough disk space (~2-3 GB for all models)
- Try restarting the Streamlit app

**Slow performance?**
- First run will be slower (model download)
- Subsequent runs use cached models
- Consider disabling models you don't need

**Memory errors?**
- Close other applications
- Try using one model at a time
- Restart the Streamlit app

