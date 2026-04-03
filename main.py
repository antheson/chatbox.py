import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from difflib import get_close_matches

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")
st.title("🛍️ ShopAssist AI - Recommendation Chatbot")

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.caption("💡 Try: cheap electronics under 100")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("diversified_ecommerce_dataset.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")

# -----------------------------
# TYPO FUNCTIONS
# -----------------------------
def correct_typo(word, word_list, cutoff=0.7):
    matches = get_close_matches(word, word_list, n=1, cutoff=cutoff)
    return matches[0] if matches else word

def correct_input(text):
    categories = [c.lower() for c in df['category'].dropna().unique()]
    words = text.lower().split()
    corrected = [correct_typo(w, categories) for w in words]
    return " ".join(corrected)

# -----------------------------
# TRAIN MODEL
# -----------------------------
training_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("help me", "help"),
    ("what can you do", "help"),
    ("recommend product", "recommend"),
    ("cheap products", "cheap"),
    ("best products", "best"),
    ("top products", "best"),
    ("discount items", "discount")
]

X = [x[0] for x in training_data]
y = [x[1] for x in training_data]

vectorizer = CountVectorizer()
model = LogisticRegression()
model.fit(vectorizer.fit_transform(X), y)

def predict_intent(text):
    return model.predict(vectorizer.transform([text]))[0]

# -----------------------------
# UI HELPERS
# -----------------------------
def show_examples():
    st.markdown("""
### 💡 You can try:
- cheap electronics under 100  
- best products  
- discount items  
- show me 5 clothing  
- give me 3 products  
""")

def display_products(data):
    st.subheader("🏆 Top Recommendations")

    for i, (_, row) in enumerate(data.iterrows(), 1):
        with st.container():
            st.markdown(f"### #{i} 🛍️ {row.get('product_name','Unknown')}")
            col1, col2, col3 = st.columns(3)

            col1.write(f"📂 {row.get('category','N/A')}")
            if 'price' in row:
                col2.write(f"💰 ${row.get('price')}")
            if 'popularity_index' in row:
                col3.write(f"⭐ {row.get('popularity_index')}")
            elif 'discount' in row:
                col3.write(f"🔥 {row.get('discount')}%")

            st.divider()

# -----------------------------
# RESPONSE ENGINE
# -----------------------------
def get_response(user_input):

    text = correct_input(user_input.lower())
    intent = predict_intent(text)

    # Greeting
    if intent == "greeting":
        return {"msg": "Hi 👋 I can recommend products for you!", "data": "EXAMPLE"}

    if intent == "help":
        return {"msg": "Here’s what you can ask 😊", "data": "EXAMPLE"}

    # Default settings
    limit = 5
    for w in text.split():
        if w.isdigit():
            limit = min(int(w), 10)

    # Category
    category = None
    for cat in df['category'].dropna().unique():
        if cat.lower() in text:
            category = cat

    # Price
    price_limit = None
    words = text.split()
    for i, w in enumerate(words):
        if w.isdigit() and i > 0 and words[i-1] in ["under","below"]:
            price_limit = float(w)

    # Intent override (important)
    if "cheap" in text:
        intent = "cheap"
    elif "best" in text or "top" in text or "popular" in text:
        intent = "best"
    elif "discount" in text:
        intent = "discount"
    elif "give me" in text or "show me" in text or "products" in text:
        intent = "recommend"

    result = df.copy()

    if category:
        result = result[result['category'].str.lower() == category.lower()]

    if price_limit:
        result = result[result['price'] <= price_limit]

    if result.empty:
        return {"msg": "No products found 😢 Try again!", "data": None}

    if intent == "cheap":
        result = result.sort_values(by='price').head(limit)
        return {"msg": f"Top {limit} cheapest products 💰", "data": result}

    elif intent == "best":
        result = result.sort_values(by='popularity_index', ascending=False).head(limit)
        return {"msg": f"Top {limit} popular products ⭐", "data": result}

    elif intent == "discount":
        result = result.sort_values(by='discount', ascending=False).head(limit)
        return {"msg": f"Top {limit} discounted products 🔥", "data": result}

    else:
        result = result.sort_values(by='popularity_index', ascending=False).head(limit)
        return {"msg": f"Top {limit} recommended products 👍", "data": result}

# -----------------------------
# CHAT SYSTEM
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        content = m["content"]
        
        # FIX: Check if content is a dictionary or string
        if isinstance(content, dict):
            # Assistant message (dictionary format)
            st.write(content["msg"])
            
            if content.get("data") == "EXAMPLE":
                show_examples()
            elif isinstance(content.get("data"), pd.DataFrame):
                display_products(content["data"])
        else:
            # User message (string format)
            st.write(content)

# Input
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})

    with st.chat_message("user"):
        st.write(user_input)

    res = get_response(user_input)

    with st.chat_message("assistant"):
        st.write(res["msg"])

        if res["data"] == "EXAMPLE":
            show_examples()
        elif isinstance(res["data"], pd.DataFrame):
            display_products(res["data"])

    st.session_state.messages.append({"role":"assistant","content":res})

    # LIMIT HISTORY (IMPORTANT)
    if len(st.session_state.messages) > 15:
        st.session_state.messages = st.session_state.messages[-15:]

    st.rerun()
