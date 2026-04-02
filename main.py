import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")
st.title("🛍️ ShopAssist AI - Recommendation Chatbot")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("diversified_ecommerce_dataset.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")

# -----------------------------
# TRAINING DATA (INTENTS)
# -----------------------------
training_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),

    ("recommend product", "recommend"),
    ("suggest something", "recommend"),

    ("cheap products", "cheap"),
    ("low price items", "cheap"),

    ("best products", "best"),
    ("top products", "best"),

    ("discount items", "discount")
]

X = [x[0] for x in training_data]
y = [x[1] for x in training_data]

vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

# -----------------------------
# INTENT PREDICTION
# -----------------------------
def predict_intent(text):
    return model.predict(vectorizer.transform([text.lower()]))[0]

# -----------------------------
# RECOMMENDATION ENGINE
# -----------------------------
def recommend_products(user_input):
    text = user_input.lower()

    # Extract price
    price_limit = None
    for word in text.split():
        if word.isdigit():
            price_limit = float(word)

    # Extract category
    category = None
    for cat in df['category'].dropna().unique():
        if cat.lower() in text:
            category = cat

    result = df.copy()

    if category:
        result = result[result['category'].str.lower() == category.lower()]

    if price_limit:
        result = result[result['price'] <= price_limit]

    return result

# -----------------------------
# RESPONSE GENERATION (NATURAL)
# -----------------------------
def get_response(user_input):
    intent = predict_intent(user_input)

    # Greeting
    if intent == "greeting":
        return "Hi there! 😊 Tell me what kind of product you're looking for — cheap, best, or discounted!"

    # Recommendation base
    result = recommend_products(user_input)

    if result.empty:
        return "Hmm, I couldn't find anything matching your request. Try something like 'cheap electronics under 100'."

    # Cheap
    if intent == "cheap":
        result = result.sort_values(by='price').head(5)
        return "Here are some budget-friendly picks for you 💰", result[['product_name', 'category', 'price']]

    # Best
    elif intent == "best":
        result = result.sort_values(by='popularity_index', ascending=False).head(5)
        return "These are the most popular products right now ⭐", result[['product_name', 'category', 'popularity_index']]

    # Discount
    elif intent == "discount":
        result = result.sort_values(by='discount', ascending=False).head(5)
        return "Check out these great deals 🔥", result[['product_name', 'category', 'discount']]

    # General recommendation
    else:
        result = result.sort_values(by='popularity_index', ascending=False).head(5)
        return "Here are some products you might like 👍", result[['product_name', 'category', 'price']]

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
            st.write(msg["content"])
        else:
            st.dataframe(msg["content"])

# Input
user_input = st.chat_input("Ask for recommendations...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input)

    with st.chat_message("assistant"):
        if isinstance(response, tuple):
            text, data = response
            st.write(text)
            st.dataframe(data)
            st.session_state.messages.append({"role": "assistant", "content": text})
            st.session_state.messages.append({"role": "assistant", "content": data})
        else:
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
