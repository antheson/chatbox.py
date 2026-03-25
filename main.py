# Install required libraries first:
# pip install nltk scikit-learn

import random
import nltk
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# -----------------------------
# e(i) DATASET (FAQs + Intents)
# -----------------------------
data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),
    ("bye", "goodbye"),
    ("see you", "goodbye"),
    ("thanks", "thanks"),
    ("thank you", "thanks"),

    ("where is my order", "track_order"),
    ("track my order", "track_order"),
    ("order status", "track_order"),

    ("i want refund", "refund"),
    ("return item", "refund"),
    ("refund my order", "refund"),

    ("recommend a phone", "recommend"),
    ("suggest laptop", "recommend"),
    ("best product for me", "recommend"),

    ("what is your service", "faq"),
    ("what do you do", "faq"),
    ("help me", "faq")
]

# Split dataset
X = [d[0] for d in data]
y = [d[1] for d in data]

# -----------------------------
# e(ii) PREPROCESSING
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

X_processed = [preprocess(x) for x in X]

# -----------------------------
# e(iii) TRAIN MODEL
# -----------------------------
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X_processed)

model = LogisticRegression()
model.fit(X_vector, y)

# -----------------------------
# d. CHATBOT FUNCTIONALITIES
# -----------------------------
def chatbot_response(user_input):
    processed = preprocess(user_input)
    vector = vectorizer.transform([processed])
    intent = model.predict(vector)[0]

    # FAQ
    if intent == "faq":
        return "I help with orders, refunds, and product recommendations."

    # Natural Conversation
    elif intent == "greeting":
        return random.choice(["Hello!", "Hi there!", "Hey! How can I help?"])

    elif intent == "goodbye":
        return "Goodbye! Have a nice day!"

    elif intent == "thanks":
        return "You're welcome!"

    # Order Tracking (Troubleshooting)
    elif intent == "track_order":
        return "Please provide your order ID."

    # Refund Logic
    elif intent == "refund":
        return "Your refund request is submitted. It will be processed in 3-5 days."

    # Recommendation
    elif intent == "recommend":
        return "Based on your needs, I recommend a mid-range smartphone or laptop."

    else:
        return "Sorry, I don't understand."

# -----------------------------
# f. TESTING
# -----------------------------
test_inputs = [
    "hello",
    "track my order",
    "i want refund",
    "recommend laptop",
    "bye"
]

print("=== Chatbot Testing ===")
for text in test_inputs:
    print(f"User: {text}")
    print(f"Bot: {chatbot_response(text)}\n")

# -----------------------------
# g. EVALUATION
# -----------------------------
y_pred = model.predict(X_vector)

print("=== Classification Report ===")
print(classification_report(y, y_pred))
