import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")
st.title("🛍️ ShopAssist AI Chatbot")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("diversified_ecommerce_dataset.csv")

# Clean column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# -----------------------------
# ML TRAINING DATA (INTENTS)
# -----------------------------
training_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),

    ("cheap product", "cheap"),
    ("low price items", "cheap"),
    ("budget products", "cheap"),

    ("best product", "best"),
    ("top items", "best"),
    ("most popular", "best"),

    ("discount products", "discount"),
    ("any discount", "discount"),

    ("electronics", "category"),
    ("clothing", "category"),
    ("beauty products", "category"),

    ("products in asia", "location"),
    ("items in europe", "location"),

    ("low stock", "stock"),
    ("stock level", "stock")
]

X = [item[0] for item in training_data]
y = [item[1] for item in training_data]

# -----------------------------
# TRAIN MODEL
# -----------------------------
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

# -----------------------------
# INTENT PREDICTION
# -----------------------------
def predict_intent(user_input):
    vec = vectorizer.transform([user_input.lower()])
    return model.predict(vec)[0]

# -----------------------------
# CHATBOT RESPONSE
# -----------------------------
def get_response(user_input):
    intent = predict_intent(user_input)

    # Greeting
    if intent == "greeting":
        return "Hello! 👋 How can I help you today?"

    # Cheap products
    elif intent == "cheap":
        result = df[df['price'] < df['price'].mean()].sort_values(by='price').head(3)
        return result[['product_name', 'price']]

    # Best products
    elif intent == "best":
        result = df.sort_values(by='popularity_index', ascending=False).head(3)
        return result[['product_name', 'popularity_index']]

    # Discount products
    elif intent == "discount":
        result = df.sort_values(by='discount', ascending=False).head(3)
        return result[['product_name', 'discount']]

    # Category search
    elif intent == "category":
        for cat in df['category'].dropna().unique():
            if cat.lower() in user_input.lower():
                result = df[df['category'].str.lower() == cat.lower()].head(3)
                return result[['product_name', 'category', 'price']]
        return "Please specify a valid category (e.g., electronics, clothing)."

    # Location-based
    elif intent == "location":
        if "in" in user_input:
            location = user_input.split("in")[-1].strip()
            result = df[df['customer_location'].str.contains(location, case=False, na=False)]
            if not result.empty:
                return result[['product_name', 'customer_location']].head(3)
        return "Please specify a location (e.g., 'products in Asia')."

    # Stock alert
    elif intent == "stock":
        result = df[df['stock_level'] < 20].head(3)
        return result[['product_name', 'stock_level']]

    else:
        return "Sorry, I didn't understand. Try asking about products 😊"

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
            st.write(msg["content"])
        else:
            st.dataframe(msg["content"])

# User input
user_input = st.chat_input("Ask me about products...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Get response
    response = get_response(user_input)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display bot response
    with st.chat_message("assistant"):
        if isinstance(response, str):
            st.write(response)
        else:
            st.dataframe(response)
