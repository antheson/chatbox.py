import pandas as pd
import streamlit as st
import nltk
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download once
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("diversified_ecommerce_dataset.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")

# -----------------------------
# ML Training Data (INTENTS)
# -----------------------------
data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),

    ("cheap product", "cheap"),
    ("low price items", "cheap"),

    ("best product", "best"),
    ("top items", "best"),

    ("discount products", "discount"),
    ("any discount", "discount"),

    ("show electronics", "category"),
    ("find clothes", "category"),

    ("products in asia", "location"),
    ("items in europe", "location"),

    ("stock level", "stock"),
    ("low stock items", "stock")
]

X = [d[0] for d in data]
y = [d[1] for d in data]

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

X_processed = [preprocess(x) for x in X]

vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X_processed)

model = LogisticRegression()
model.fit(X_vector, y)

# -----------------------------
# Predict Intent
# -----------------------------
def predict_intent(user_input):
    processed = preprocess(user_input)
    vec = vectorizer.transform([processed])
    return model.predict(vec)[0]

# -----------------------------
# Chatbot Logic
# -----------------------------
def get_response(user_input):
    intent = predict_intent(user_input)

    if intent == "greeting":
        return "Hello! How can I help you today? 😊"

    elif intent == "cheap":
        cheap = df[df['price'] < df['price'].mean()]
        return cheap[['product_name', 'price']].head(3)

    elif intent == "best":
        best = df.sort_values(by='popularity_index', ascending=False).head(3)
        return best[['product_name', 'popularity_index']]

    elif intent == "discount":
        discount = df.sort_values(by='discount', ascending=False).head(3)
        return discount[['product_name', 'discount']]

    elif intent == "category":
        for cat in df['category'].str.lower().unique():
            if cat in user_input.lower():
                results = df[df['category'].str.lower() == cat]
                return results[['product_name', 'price']].head(3)
        return "Please specify a category."

    elif intent == "location":
        location = user_input.split("in")[-1].strip()
        results = df[df['customer_location'].str.contains(location, case=False, na=False)]
        if not results.empty:
            return results[['product_name', 'customer_location']].head(3)
        return "No results found for that location."

    elif intent == "stock":
        low = df[df['stock_level'] < 20]
        return low[['product_name', 'stock_level']].head(3)

    else:
        return "Sorry, I didn’t understand."

# -----------------------------
# Streamlit Chat UI
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")

st.title("🛍️ ShopAssist AI Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
            st.write(msg["content"])
        else:
            st.dataframe(msg["content"])

# User input
user_input = st.chat_input("Ask me anything about products...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user
    with st.chat_message("user"):
        st.write(user_input)

    # Bot response
    response = get_response(user_input)

    # Save bot message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display bot
    with st.chat_message("assistant"):
        if isinstance(response, str):
            st.write(response)
        else:
            st.dataframe(response)
