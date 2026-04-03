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

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("diversified_ecommerce_dataset.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")

# -----------------------------
# TYPO CORRECTION FUNCTION
# -----------------------------
def correct_typo(word, word_list, cutoff=0.8):
    """Correct typo in a word using difflib"""
    if word in word_list:
        return word
    
    matches = get_close_matches(word, word_list, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    return word

def correct_category_typo(user_input, categories):
    """Check and correct category typos in user input"""
    words = user_input.lower().split()
    corrected_words = []
    
    for word in words:
        # Check if this word might be a category (length > 3 to avoid correcting small words)
        if len(word) > 3:
            corrected = correct_typo(word, categories, cutoff=0.7)
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

def correct_intent_typo(user_input):
    """Correct typos in intent-related keywords"""
    intent_keywords = {
        'hello': ['hello', 'hllo', 'helo', 'hellp', 'hallow', 'halo'],
        'help': ['help', 'halp', 'hlp', 'hepl', 'helpp'],
        'cheap': ['cheap', 'cheep', 'chap', 'chep', 'cheapp'],
        'best': ['best', 'bests', 'besst', 'bist', 'bested'],
        'discount': ['discount', 'discont', 'dicount', 'discout', 'diskaunt'],
        'categories': ['categories', 'catagories', 'categries', 'catgories', 'categorys'],
        'recommend': ['recommend', 'recomend', 'reccomend', 'rekomend', 'recommanded']
    }
    
    words = user_input.lower().split()
    corrected_words = []
    
    for word in words:
        corrected = word
        for keyword, variations in intent_keywords.items():
            if word in variations:
                corrected = keyword
                break
        corrected_words.append(corrected)
    
    return ' '.join(corrected_words)

# -----------------------------
# TRAINING DATA (INTENTS)
# -----------------------------
training_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),
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
    ("discount items", "discount"),
    ("show categories", "categories"),
    ("what categories", "categories"),
    ("list categories", "categories")
]

X = [x[0] for x in training_data]
y = [x[1] for x in training_data]

vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

# -----------------------------
# INTENT PREDICTION WITH TYPO HANDLING
# -----------------------------
def predict_intent(text):
    # First correct common typos in the input
    corrected_text = correct_intent_typo(text)
    return model.predict(vectorizer.transform([corrected_text.lower()]))[0]

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
- show all categories  
- what categories do you have?

### 🔤 Typo-friendly! Try:
- cheep products (understands "cheap")
- rekomend something (understands "recommend")
- catagories (understands "categories")
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
# DISPLAY CATEGORIES (UI)
# -----------------------------
def display_categories():
    """Display all available product categories"""
    categories = df['category'].dropna().unique()
    categories = sorted(categories)
    
    st.subheader("📚 Available Product Categories")
    
    # Create columns for better display
    cols = st.columns(3)
    for idx, category in enumerate(categories):
        with cols[idx % 3]:
            # Count products in each category
            product_count = len(df[df['category'] == category])
            st.write(f"• **{category}** ({product_count} products)")
    
    st.info(f"💡 Total: {len(categories)} categories available")
    
    # Optional: Show sample products from random category
    with st.expander("🔍 Want to see sample products from a category?"):
        selected_category = st.selectbox("Choose a category:", categories)
        if selected_category:
            sample_products = df[df['category'] == selected_category].head(3)
            st.write(f"**Sample products in {selected_category}:**")
            for _, product in sample_products.iterrows():
                st.write(f"• {product['product_name']} - ${product.get('price', 'N/A')}")

# -----------------------------
# RESPONSE GENERATION WITH TYPO HANDLING
# -----------------------------
def get_response(user_input):
    # Apply typo correction to the entire input
    categories_list = [cat.lower() for cat in df['category'].dropna().unique()]
    
    # Correct category typos first
    corrected_input = correct_category_typo(user_input, categories_list)
    
    # Also correct intent typos
    corrected_input = correct_intent_typo(corrected_input)
    
    text = corrected_input.lower()

    # Predict intent
    intent = predict_intent(corrected_input)
    
    # Check for category intent first
    if any(phrase in text for phrase in ["show categories", "what categories", "list categories", "all categories", "available categories", "catagories"]):
        intent = "categories"

    # Greeting
    if intent == "greeting":
        return {
            "type": "text",
            "message": "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, or popularity!\n\n💡 Tip: I understand typos too!",
            "data": "SHOW_EXAMPLES"
        }

    # Help
    if intent == "help":
        return {
            "type": "text",
            "message": "Here are some things you can ask me 😊",
            "data": "SHOW_EXAMPLES"
        }
    
    # Show categories
    if intent == "categories":
        return {
            "type": "categories",
            "message": "Here are all the product categories available in our store! 🛍️",
            "data": None
        }

    # -----------------------------
    # DEFAULT SETTINGS
    # -----------------------------
    limit = 5

    for word in text.split():
        if word.isdigit():
            limit = min(int(word), 10)

    # Extract category (with typo handling)
    category = None
    for cat in df['category'].dropna().unique():
        if cat.lower() in text or correct_typo(cat.lower(), text.split(), cutoff=0.7) in text:
            category = cat
            break

    # Extract price
    price_limit = None
    words = text.split()
    for i, w in enumerate(words):
        if w.isdigit():
            if i > 0 and words[i-1] in ["under", "below", "less", "than"]:
                price_limit = float(w)

    # -----------------------------
    # INTENT (HYBRID FIX WITH TYPO SUPPORT)
    # -----------------------------
    if "cheap" in text or "cheep" in text or "chap" in text:
        intent = "cheap"
    elif "best" in text or "bests" in text or "besst" in text:
        intent = "best"
    elif "discount" in text or "discont" in text or "dicount" in text:
        intent = "discount"
    elif "recommend" in text or "recomend" in text or "rekomend" in text or "show me" in text or "give me" in text:
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
        return {
            "type": "text",
            "message": "I couldn't find matching products. Try changing your filters or check for typos! 😊",
            "data": None
        }

    # -----------------------------
    # RECOMMENDATION LOGIC
    # -----------------------------
    if intent == "cheap":
        result = result.sort_values(by='price').head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are {limit} budget-friendly products 💰",
            "data": result[['product_name','category','price']]
        }

    elif intent == "best":
        result = result.sort_values(by='popularity_index', ascending=False).head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are the top {limit} most popular products ⭐",
            "data": result[['product_name','category','popularity_index']]
        }

    elif intent == "discount":
        result = result.sort_values(by='discount', ascending=False).head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are the top {limit} discounted products 🔥",
            "data": result[['product_name','category','discount']]
        }

    else:
        result = result.sort_values(by='popularity_index', ascending=False).head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are {limit} recommended products 👍",
            "data": result[['product_name','category','price']]
        }

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        
        # Check if content is a dictionary (our new format)
        if isinstance(content, dict):
            # Display the message
            st.write(content["message"])
            
            # Handle the data based on its type
            data_value = content["data"]
            response_type = content.get("type", "")
            
            # IMPORTANT: Check type BEFORE comparing
            if response_type == "categories":
                display_categories()
            elif data_value is not None:
                if isinstance(data_value, str) and data_value == "SHOW_EXAMPLES":
                    show_examples()
                elif isinstance(data_value, pd.DataFrame):
                    display_products(data_value, label="Top Recommendations")
        else:
            # Handle old string format
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
        if isinstance(response, dict):
            # Display the message
            st.write(response["message"])
            
            # Handle the data based on its type
            data_value = response["data"]
            response_type = response.get("type", "")
            
            # IMPORTANT: Check type BEFORE comparing
            if response_type == "categories":
                display_categories()
            elif data_value is not None:
                if isinstance(data_value, str) and data_value == "SHOW_EXAMPLES":
                    show_examples()
                elif isinstance(data_value, pd.DataFrame):
                    display_products(data_value, label="Top Recommendations")
            
            # Store the response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
        else:
            # Fallback for any other response format
            st.write(str(response))
            st.session_state.messages.append({
                "role": "assistant",
                "content": str(response)
            })
