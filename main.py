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
# HELP WITH EXAMPLES QUES
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
    text = user_input.lower()

    # Greeting
    if intent == "greeting":
        return "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, or popularity!", "SHOW_EXAMPLES"

    # Help intent
    elif intent == "help":
        return "Here are some things you can ask me 😊", "SHOW_EXAMPLES"

    # -----------------------------
    # DEFAULT SETTINGS
    # -----------------------------
    limit = 5

    # Extract number
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
    # DETECT INTENT (HYBRID 🔥)
    # -----------------------------
    intent = predict_intent(user_input)

    # Backup keyword detection (VERY IMPORTANT)
    if "cheap" in text or "low price" in text:
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
        return "I couldn't find matching products. Try changing your filters 😊"

    # -----------------------------
    # RECOMMENDATION LOGIC
    # -----------------------------
    if intent == "cheap":
        result = result.sort_values(by='price').head(limit)
        return f"Here are {limit} budget-friendly products 💰", result[['product_name','category','price']]

    elif intent == "best":
        result = result.sort_values(by='popularity_index', ascending=False).head(limit)
        return f"Here are the top {limit} most popular products ⭐", result[['product_name','category','popularity_index']]

    elif intent == "discount":
        result = result.sort_values(by='discount', ascending=False).head(limit)
        return f"Here are the top {limit} discounted products 🔥", result[['product_name','category','discount']]

    else:
        # DEFAULT recommendation
        result = result.sort_values(by='popularity_index', ascending=False).head(limit)
        return f"Here are {limit} recommended products 👍", result[['product_name','category','price']]
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

        if data == "SHOW_EXAMPLES":
            show_examples()
        else:
            display_products(data, label="Top Recommendations")

        st.session_state.messages.append({
            "role": "assistant",
            "content": text
        })

    else:
        st.write(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
