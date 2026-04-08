import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")
st.title("🛍️ ShopAssist AI - Adidas Recommendation Bot")

# Clear chat
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("adidas_usa.csv")

# Clean + rename columns
df.columns = df.columns.str.lower().str.replace(" ", "_")

df = df.rename(columns={
    "name": "product_name",
    "selling_price": "price",
    "average_rating": "rating"
})

# Create discount column
df["discount"] = ((df["original_price"] - df["price"]) / df["original_price"]) * 100

# -----------------------------
# TRAIN INTENT MODEL
# -----------------------------
training_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("help", "help"),
    ("cheap products", "cheap"),
    ("best products", "best"),
    ("top products", "best"),
    ("discount items", "discount"),
    ("recommend something", "recommend")
]

X = [x[0] for x in training_data]
y = [x[1] for x in training_data]

vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

def predict_intent(text):
    return model.predict(vectorizer.transform([text.lower()]))[0]

# -----------------------------
# DISPLAY PRODUCTS UI
# -----------------------------
def display_products(df_result):
    if df_result.empty:
        st.warning("No products found.")
        return

    st.subheader("🏆 Top Recommendations")

    for i, (_, row) in enumerate(df_result.iterrows(), start=1):
        with st.container():
            st.markdown(f"### #{i} 🛍️ {row['product_name']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"📂 Category: {row['category']}")

            with col2:
                st.write(f"💰 Price: ${row['price']}")

            with col3:
                st.write(f"⭐ Rating: {row.get('rating', 'N/A')}")

            st.write(f"🔥 Discount: {round(row['discount'],1)}%")
            st.divider()

# -----------------------------
# RESPONSE LOGIC
# -----------------------------
def get_response(user_input):
    text = user_input.lower()
    intent = predict_intent(text)

    # Greeting
    if intent == "greeting":
        return {
            "msg": "Hi 👋 I can recommend Adidas products!\n\nTry:\n- cheap shoes under 100\n- best products\n- discount items",
            "data": None
        }

    # Help
    if intent == "help":
        return {
            "msg": "You can ask:\n- cheap products\n- best products\n- discount items\n- shoes under 100",
            "data": None
        }

    # -----------------------------
    # EXTRACT CONDITIONS
    # -----------------------------
    limit = 5
    price_limit = None
    category = None

    # Extract number
    for word in text.split():
        if word.isdigit():
            limit = min(int(word), 10)

    # Extract price
    words = text.split()
    for i, w in enumerate(words):
        if w.isdigit() and i > 0 and words[i-1] in ["under", "below"]:
            price_limit = float(w)

    # Extract category
    for cat in df['category'].dropna().unique():
        if cat.lower() in text:
            category = cat
            break

    # Handle unclear request
    if "products" in text and not category:
        return {
            "msg": "What type of products are you looking for? 😊\n\nExample:\n- shoes under 100\n- cheap clothing",
            "data": None
        }

    # -----------------------------
    # FILTER DATA
    # -----------------------------
    result = df.copy()

    if category:
        result = result[result['category'].str.lower() == category.lower()]

    if price_limit:
        result = result[result['price'] <= price_limit]

    if result.empty:
        return {
            "msg": "No matching products found. Try different filters 😊",
            "data": None
        }

    # -----------------------------
    # RECOMMENDATION TYPES
    # -----------------------------
    if "cheap" in text or intent == "cheap":
        result = result.sort_values(by='price').head(limit)

    elif "best" in text or "top" in text or intent == "best":
        result = result.sort_values(by=['rating','reviews_count'], ascending=False).head(limit)

    elif "discount" in text or intent == "discount":
        result = result.sort_values(by='discount', ascending=False).head(limit)

    else:
        result = result.sort_values(by=['rating','reviews_count'], ascending=False).head(limit)

    return {
        "msg": f"Here are top {limit} products 👍",
        "data": result
    }

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"]["msg"])
        if msg["content"]["data"] is not None:
            display_products(msg["content"]["data"])

# Input
user_input = st.chat_input("Ask me about Adidas products...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": {"msg": user_input, "data": None}
    })

    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input)

    with st.chat_message("assistant"):
        st.write(response["msg"])
        if response["data"] is not None:
            display_products(response["data"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()
