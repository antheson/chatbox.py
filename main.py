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
    text = user_input.lower()

    # Greeting
    if intent == "greeting":
        return "Hello! 👋 I can recommend products based on price, category, or popularity!"

    # -----------------------------
    # SMART RECOMMENDATION ENGINE 🔥
    # -----------------------------

    # Extract price condition
    price_limit = None
    words = text.split()
    for i, w in enumerate(words):
        if w.isdigit():
            price_limit = float(w)

    # Extract category
    selected_category = None
    for cat in df['category'].dropna().unique():
        if cat.lower() in text:
            selected_category = cat

    # Base dataset
    result = df.copy()

    # Apply filters
    if selected_category:
        result = result[result['category'].str.lower() == selected_category.lower()]

    if price_limit:
        result = result[result['price'] <= price_limit]

    # -----------------------------
    # Recommendation Types
    # -----------------------------

    # Cheap recommendation
    if "cheap" in text or intent == "cheap":
        result = result.sort_values(by='price').head(5)
        return result[['product_name', 'category', 'price']]

    # Best recommendation
    elif "best" in text or "top" in text or intent == "best":
        result = result.sort_values(by='popularity_index', ascending=False).head(5)
        return result[['product_name', 'category', 'popularity_index']]

    # Discount recommendation
    elif "discount" in text or intent == "discount":
        result = result.sort_values(by='discount', ascending=False).head(5)
        return result[['product_name', 'category', 'discount']]

    # Combined recommendation (A+)
    elif selected_category or price_limit:
        result = result.sort_values(by='popularity_index', ascending=False).head(5)
        return result[['product_name', 'category', 'price', 'popularity_index']]

    # Location-based
    elif intent == "location":
        if "in" in text:
            location = text.split("in")[-1].strip()
            result = df[df['customer_location'].str.contains(location, case=False, na=False)]
            return result[['product_name', 'customer_location']].head(5)

    # Stock alert
    elif intent == "stock":
        result = df[df['stock_level'] < 20].head(5)
        return result[['product_name', 'stock_level']]

    # Default
    else:
        return "Try asking:\n- cheap electronics\n- best products under 100\n- discount items\n- products in Asia"
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
