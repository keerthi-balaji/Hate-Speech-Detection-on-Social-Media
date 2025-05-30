# Hate-Speech-Detection-on-Social-Media
A binary text classification project using CNN, GloVe-enhanced CNN, and fine-tuned BERT for detecting hate speech online. (NLP Course Project)

## Overview
This project explores automated detection of hate speech in social media text, utilizing deep learning and NLP models:
- A custom CNN model
- A GloVe-enhanced CNN with LSTM
- A fine-tuned BERT Transformer

**Best performance: Fine-tuned BERT achieved 93.50% accuracy and 93.40% F1-score, outperforming both CNN models.**

## Problem Statement
Hate speech online poses serious risks to targeted communities and individuals. Manual detection is infeasible at scale due to sarcasm, slang, and contextual subtleties. This project aims to build a robust, scalable hate speech classifier using state-of-the-art machine learning techniques.

## Dataset
The "Curated Hate Speech Dataset" comprises a total of 451,709 sentences, with 371,452 labeled as hate speech and 80,250 as non-hate speech. The data collection process involved sourcing content from diverse online platforms to compile a comprehensive set of hate speech sentences that include several forms of hateful text, including emojis, emoticons, and slang. The most prominent source is from the platform X, formerly known as Twitter.

| ClassLabel    | Count    |
|-----------------|----------|
| HateSpeech      | 371,452  |
| Non-HateSpeech  | 80,250   |

The dataset can be found publicly at https://data.mendeley.com/datasets/9sxpkmm8xn/1

## Data Preprocessing
- Removed noise: hyperlinks, mentions, emojis, emoticons
- Cleaned: contractions, grammar, special characters, numbers
- Balanced: downsampling majority class
- Tokenized: using BERT tokenizer or word indices for CNNs

## Models Implemented
### CNN (Custom Embeddings)
- Layers: Embedding → Conv1D → GlobalMaxPooling → Dense → Dropout → Sigmoid
- Accuracy: 85.95%
- F1-Score: 86.52%

### GloVe-enhanced CNN + LSTM
- Pretrained Embeddings: GloVe (Twitter)
- Architecture: Embedding → Conv1D → MaxPooling → LSTM → Dense → Sigmoid
- Accuracy: 84.25%
- F1-Score: 83.89%

### Fine-tuned Transformer (BERT)
- Backbone: Pre-trained BERT (base-uncased)
- Fine-tuning: 5 epochs, RAdam optimizer + scheduler
- Accuracy: 93.50%
- F1-Score: 93.40%

## Results Summary
|Model|Accuracy|	Precision|	Recall	|F1-Score|
|------|--------|-----------|----------|--------|
|CNN	|85.95%	|83.74%	|89.49%	|86.52%|
|GloVe-CNN + LSTM	|84.25%	|84.61%	|84.02%	|83.89%|
|Fine-tuned BERT	|93.50%	|94.50%	|92.40%	|93.40%|

## Key Insights
- CNN models are prone to overfitting, especially without pretrained embeddings.
- GloVe embeddings improved semantic understanding but lacked sequential context.
- BERT excelled due to its contextual and bidirectional representation, making it ideal for nuanced hate speech detection.

## Technologies Used
- Python, TensorFlow, PyTorch
- Scikit-learn, Pandas, NumPy
- BERT via HuggingFace Transformers
- GloVe embeddings (Stanford NLP)
- Matplotlib, Seaborn (for visualizations)

👥 Authors
Keerthi Balaji, Austin Robinson, Yashi Yadav
Department of Computer Science, Purdue University Fort Wayne
