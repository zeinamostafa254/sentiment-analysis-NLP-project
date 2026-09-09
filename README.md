# Wavelength — Bilingual Sentiment Analysis

A small NLP app that reads a product/app review in **English or Arabic**, figures out which language it's written in, and predicts whether the sentiment is **positive** or **negative**, with a confidence score.

Built as an end-to-end project: data cleaning → text preprocessing → model training → a styled Streamlit interface.

## How it works

The app is really three small models working together:

1. **Language classifier** — a logistic regression model (TF-IDF + `LogisticRegression`) trained to tell English text apart from Arabic text.
2. **English sentiment model** — trained on a movie review dataset, predicts Positive / Negative.
3. **Arabic sentiment model** — trained on Arabic app-store reviews, predicts Positive / Negative.

When you submit a review, the app:
1. Runs it through the language classifier.
2. Routes it to the matching sentiment model (English or Arabic).
3. Preprocesses the text (cleaning, stop-word removal, lemmatization for English; normalization for Arabic) before prediction.
4. Shows the detected language, the predicted sentiment, and the model's confidence.

## Project structure

```
.
├── app.py                              # Streamlit UI
├── Preprocessing_pipeline.py           # Text cleaning + language detection helpers
├── English_model.py                    # Loads English model, exposes predict_english()
├── Arabic_model.py                     # Loads Arabic model, exposes predict_arabic()
├── English_model_weights.pkl           # Trained English sentiment model + vectorizer
├── Arabic_model_weights.pkl            # Trained Arabic sentiment model + vectorizer
├── Language_classifier_weights.pkl     # Trained language classifier + vectorizer
├── Training_english_model.ipynb        # Notebook: training the English model
├── Training_arabic_model.ipynb         # Notebook: training the Arabic model
├── Training_language_classifier.ipynb  # Notebook: training the language classifier
└── data/
    ├── Final_Data.csv                  # Arabic app review dataset
    └── MovieReviewTrainingDatabase.csv # English movie review dataset
```

## Setup

**Requirements:** Python 3.10+

```bash
pip install streamlit scikit-learn joblib nltk pandas
```

The app downloads the NLTK data it needs (`punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`) automatically on first run.

## Run it

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), type a review in English or Arabic, and hit **Read the sentiment**.

## Example

| Input | Detected language | Sentiment |
|---|---|---|
| "The delivery was late but the food was amazing" | English | Positive |
| "التطبيق بطيء جدا ومزعج" | Arabic | Negative |

## Retraining the models

The three `Training_*.ipynb` notebooks contain the full training process (data cleaning, vectorization, model fitting, evaluation). Each trained model + its vectorizer is saved together as a dictionary (`{"model": ..., "vectorizer": ...}`) via `joblib`, which is what `English_model.py`, `Arabic_model.py`, and `Preprocessing_pipeline.detect_language()` load at runtime.

## Notes

- Models are classic scikit-learn (TF-IDF + Logistic Regression), not deep learning — fast to run, no GPU needed.
- Arabic preprocessing normalizes letter variants (e.g. أ/إ/آ → ا) and strips diacritics before vectorizing.
- English preprocessing keeps negation words (e.g. "not", "isn't") out of the stop-word list, since they flip sentiment.
