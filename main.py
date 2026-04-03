import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")
st.title("🛍️ ShopAssist AI - Recommendation Chatbot")

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

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

    ("what can you do", "help"),
    ("help me", "help"),
    ("how to use", "help"),
    ("what can i ask", "help"),

    ("recommend product", "recommend"),
    ("suggest something", "recommend"),

    ("cheap products", "cheap"),
    ("low price items", "cheap"),

    ("best products", "best"),
    ("top products", "best"),
    ("most popular", "best"),

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
# SHOW EXAMPLE QUESTIONS
# -----------------------------
def show_examples():
    st.markdown("""
### 💡 You can try asking:
- cheap electronics under 100  
- best products  
- discount items  
- show me 5 cheap clothing  
- recommend something  
""")

# -----------------------------
# DISPLAY PRODUCTS (UI)
# -----------------------------
def display_products(df_result, label="Recommended Products"):
    if df_result.empty:
        st.warning("No products found.")
        return

    st.subheader(f"🏆 {label}")

    for i, (_, row) in enumerate(df_result.iterrows(), start=1):
        with st.container():
            st.markdown(f"### #{i} 🛍️ {row.get('product_name', 'Unknown')}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"📂 Category: {row.get('category', 'N/A')}")

            with col2:
                if 'price' in row:
                    st.write(f"💰 Price: ${row.get('price', 'N/A')}")

            with col3:
                if 'popularity_index' in row:
                    st.write(f"⭐ Popularity: {row.get('popularity_index', 'N/A')}")
                elif 'discount' in row:
                    st.write(f"🔥 Discount: {row.get('discount', 'N/A')}%")

            st.divider()

# -----------------------------
# RESPONSE GENERATION
# -----------------------------
def get_response(user_input):
    text = user_input.lower()

    # Predict intent
    intent = predict_intent(user_input)

    # Greeting
    if intent == "greeting":
        return ("text", "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, or popularity!", "SHOW_EXAMPLES")

    # Help
    if intent == "help":
        return ("text", "Here are some things you can ask me 😊", "SHOW_EXAMPLES")

    # -----------------------------
    # DEFAULT SETTINGS
    # -----------------------------
    limit = 5

    for word in text.split():
        if word.isdigit():
            limit = min(int(word), 10)

    # Extract category
    category = None
    for cat in df['category'].dropna().unique():
        if cat.lower() in text:
            category = cat

    # Extract price
    price_limit = None
    words = text.split()
    for i, w in enumerate(words):
        if w.isdigit():
            if i > 0 and words[i-1] in ["under", "below"]:
                price_limit = float(w)

    # -----------------------------
    # INTENT (HYBRID FIX)
    # -----------------------------
    if "cheap" in text:
        intent = "cheap"
    elif "best" in text or "top" in text or "popular" in text:
        intent = "best"
    elif "discount" in text:
        intent = "discount"
    elif "recommend" in text or "show me" in text or "give me" in text:
        intent = "recommend"

    # -----------------------------
    # FILTER DATA
    # -----------------------------
    result = df.copy()

    if category:
        result = result[result['category'].str.lower() == category.lower()]

    if price_limit:
        result = result[result['price'] <= price_limit]

    if result.empty:
        return ("text", "I couldn't find matching products. Try changing your filters 😊", None)

    # -----------------------------
    # RECOMMENDATION LOGIC
    # -----------------------------
    if intent == "cheap":
        result = result.sort_values(by='price').head(limit)
        return ("dataframe", f"Here are {limit} budget-friendly products 💰", result[['product_name','category','price']])

    elif intent == "best":
        result = result.sort_values(by='popularity_index', ascending=False).head(limit)
        return ("dataframe", f"Here are the top {limit} most popular products ⭐", result[['product_name','category','popularity_index']])

    elif intent == "discount":
        result = result.sort_values(by='discount', ascending=False).head(limit)
        return ("dataframe", f"Here are the top {limit} discounted products 🔥", result[['product_name','category','discount']])

    else:
        result = result.sort_values(by='popularity_index', ascending=False).head(limit)
        return ("dataframe", f"Here are {limit} recommended products 👍", result[['product_name','category','price']])

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        
        # Handle different message formats
        if isinstance(content, dict):
            # Old format - handle gracefully
            if "text" in content:
                st.write(content["text"])
            if "data" in content and content["data"] is not None:
                if content["data"] == "SHOW_EXAMPLES":
                    show_examples()
                elif isinstance(content["data"], pd.DataFrame):
                    display_products(content["data"], label="Top Recommendations")
        elif isinstance(content, tuple) and len(content) == 3:
            # New format (type, text, data)
            msg_type, msg_text, msg_data = content
            st.write(msg_text)
            if msg_data == "SHOW_EXAMPLES":
                show_examples()
            elif isinstance(msg_data, pd.DataFrame):
                display_products(msg_data, label="Top Recommendations")
        else:
            # Just a string
            st.write(content)

# User input
user_input = st.chat_input("Ask for recommendations...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input)

    with st.chat_message("assistant"):
        if isinstance(response, tuple) and len(response) == 3:
            response_type, text, data = response
            
            st.write(text)
            
            if data == "SHOW_EXAMPLES":
                show_examples()
            elif isinstance(data, pd.DataFrame):
                display_products(data, label="Top Recommendations")
            
            # Store the response in the new format
            st.session_state.messages.append({
                "role": "assistant",
                "content": response  # Store as tuple (type, text, data)
            })
        else:
            # Fallback for any other response format
            st.write(str(response))
            st.session_state.messages.append({
                "role": "assistant",
                "content": str(response)
            })
