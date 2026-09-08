# note that this preprocessing part is found in the notebook initial_draft
import nltk
import re
# i will use word lemmatization imstead of stemming, to reach a better evaluation
# i will remove the stop words but without the negative words because it affects the meaning

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

stop_words = set(stopwords.words("english"))
negation_words = {
    "no", "not", "nor", "never","don't", "doesn't", "didn't","isn't", "wasn't", "weren't",
    "won't", "wouldn't", "can't", "couldn't", "shouldn't"
}

stop_words = stop_words - negation_words


lemmatizer = WordNetLemmatizer()

# part of speech tagging for lemmatization
def get_wordnet_pos(word):

    tag = nltk.pos_tag([word])[0][1]

    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# english preprocessing pipeline
def preprocess_english(text):

    text = str(text)
    # Lowercase
    text = text.lower()
    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove punctuation/numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Tokenization
    tokens = word_tokenize(text)
    # Stop-word removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization with part of speech tagging
    tokens = [
        lemmatizer.lemmatize(
            word,
            get_wordnet_pos(word)
        )
        for word in tokens
    ]

    return " ".join(tokens)

# arabic preprocessing pipeline (same idea)
# searched for it
def normalize_arabic(text):

    # Alef normalization
    text = re.sub("[إأآا]", "ا", text)

    # Ya normalization
    text = re.sub("ى", "ي", text)

    # Remove diacritics
    text = re.sub(
        r"[\u0617-\u061A\u064B-\u0652]",
        "",
        text
    )

    # Remove Tatweel
    text = text.replace("ـ", "")

    return text


def preprocess_arabic(text):

    text = str(text)

    # Normalize Arabic
    text = normalize_arabic(text)
    # Remove URLs
    text = re.sub( r"http\S+|www\S+"," ",text)
    # Keep Arabic characters
    text = re.sub(r"[^\u0600-\u06FF\s]"," ",text)
    # Remove extra spaces
    text = re.sub(r"\s+"," ",text)

    return text.strip()

def detect_language(text, language_model, language_vectorizer):
    vector = language_vectorizer.transform([text])

    prediction = language_model.predict(vector)[0]

    return prediction