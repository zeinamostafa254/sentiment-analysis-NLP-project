import joblib

from Preprocessing_pipeline import preprocess_arabic


# Load trained Arabic model and vectorizer
weights = joblib.load("Arabic_model_weights.pkl")

model = weights["model"]
vectorizer = weights["vectorizer"]


def predict_arabic(text):
    """
    Predict sentiment of an Arabic review.

    Returns:
        sentiment: Positive or Negative
        confidence: model confidence
    """

    cleaned_text = preprocess_arabic(text)

    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = max(probabilities)

    return sentiment, confidence
