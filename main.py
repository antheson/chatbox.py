import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from difflib import get_close_matches

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")
st.title("🛍️ ShopAssist AI - Adidas Recommendation Chatbot")

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("adidas_usa.csv")
    # Rename columns to match expected format
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Rename specific columns to match the code expectations
    column_mapping = {
        'name': 'product_name',
        'selling_price': 'price',
        'original_price': 'original_price',
        'average_rating': 'popularity_index',
        'reviews_count': 'review_count',
        'color': 'color',
        'sku': 'sku',
        'availability': 'availability'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Handle price - if selling_price is empty, use original_price
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'original_price' in df.columns:
        df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
    
    # Fill missing prices with original_price
    if 'price' in df.columns and 'original_price' in df.columns:
        df['price'] = df['price'].fillna(df['original_price'])
    
    # Clean price column
    df['price'] = df['price'].fillna(0)
    
    # Create discount column if not exists (calculate from original_price and selling_price)
    if 'original_price' in df.columns and 'price' in df.columns:
        df['discount'] = ((df['original_price'] - df['price']) / df['original_price'] * 100).fillna(0)
        df['discount'] = df['discount'].round(0).astype(int)
    else:
        df['discount'] = 0
    
    # Fill NaN values
    df['product_name'] = df['product_name'].fillna('Unknown Product')
    df['category'] = df['category'].fillna('Uncategorized')
    df['color'] = df['color'].fillna('Multiple Colors')
    
    return df

df = load_data()

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
    corrected_text = correct_intent_typo(text)
    return model.predict(vectorizer.transform([corrected_text.lower()]))[0]

# -----------------------------
# SHOW EXAMPLE QUESTIONS
# -----------------------------
def show_examples():
    st.markdown("""
### 💡 You can try asking:
- cheap shoes under 100  
- best products  
- discount items  
- show me 5 cheap clothing  
- recommend something  
- show all categories  
- what categories do you have?
""")

# -----------------------------
# DISPLAY PRODUCTS (UI) - SIMPLIFIED VERSION
# -----------------------------
def display_products(df_result, label="Recommended Products"):
    if df_result.empty:
        st.warning("No products found.")
        return

    st.subheader(f"🏆 {label}")

    for i, (_, row) in enumerate(df_result.iterrows(), start=1):
        with st.container():
            # Create a card-like appearance
            st.markdown(f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                background-color: #fafafa;
            ">
                <h3 style="margin-top: 0;">#{i} 🛍️ {row.get('product_name', 'Unknown')}</h3>
            """, unsafe_allow_html=True)
            
            # Create 5 columns for key information
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.write(f"📂 **Category:** {row.get('category', 'N/A')}")
            
            with col2:
                if 'color' in row:
                    st.write(f"🎨 **Color:** {row.get('color', 'N/A')}")
            
            with col3:
                if 'price' in row and row['price'] > 0:
                    st.write(f"💰 **Price:** ${row['price']:.2f}")
            
            with col4:
                if 'popularity_index' in row and row['popularity_index'] > 0:
                    rating = row['popularity_index']
                    stars = "⭐" * int(round(rating)) + "☆" * (5 - int(round(rating)))
                    st.write(f"⭐ **Rating:** {stars} ({rating}/5)")
            
            with col5:
                if 'discount' in row and row['discount'] > 0:
                    st.write(f"🔥 **Discount:** {row['discount']}% OFF")
                elif 'review_count' in row and row['review_count'] > 0:
                    st.write(f"📝 **Reviews:** {int(row['review_count'])}")
            
            # Show original price if available and different
            if 'original_price' in row and row['original_price'] > row['price']:
                st.caption(f"~~Original: ${row['original_price']:.2f}~~")
            
            st.markdown("</div>", unsafe_allow_html=True)

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
            product_count = len(df[df['category'] == category])
            # Get unique colors in this category
            unique_colors = df[df['category'] == category]['color'].nunique()
            st.write(f"• **{category}** ({product_count} products, {unique_colors} colors)")
    
    st.info(f"💡 Total: {len(categories)} categories available")
    
    # Show sample products from random category
    with st.expander("🔍 Want to see sample products from a category?"):
        selected_category = st.selectbox("Choose a category:", categories)
        if selected_category:
            sample_products = df[df['category'] == selected_category].head(5)
            st.write(f"**Sample products in {selected_category}:**")
            for _, product in sample_products.iterrows():
                color_info = f" ({product['color']})" if product['color'] != 'Multiple Colors' else ""
                st.write(f"• {product['product_name']}{color_info} - ${product.get('price', 0):.2f}")

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
            "message": "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, color, or rating!",
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

    # Extract color (if mentioned)
    color = None
    colors_list = df['color'].dropna().unique()
    for col in colors_list:
        if col.lower() in text:
            color = col
            break

    # Extract price
    price_limit = None
    words = text.split()
    for i, w in enumerate(words):
        if w.isdigit():
            if i > 0 and words[i-1] in ["under", "below", "less", "than"]:
                price_limit = float(w)

    # -----------------------------
    # INTENT DETECTION
    # -----------------------------
    if "cheap" in text or "cheep" in text or "chap" in text:
        intent = "cheap"
    elif "best" in text or "bests" in text or "besst" in text:
        intent = "best"
    elif "discount" in text or "discont" in text or "dicount" in text:
        intent = "discount"
    elif "recommend" in text or "recomend" in text or "rekomend" in text or "show me" in text or "give me" in text:
        intent = "recommend"
    elif "products" in text and not category:
        return {
            "type": "text",
            "message": "Sure! 😊 What type of products are you looking for?\n\nYou can say:\n- cheap shoes\n- best clothing\n- products under 100",
            "data": "SHOW_EXAMPLES"
        }

    # -----------------------------
    # FILTER DATA
    # -----------------------------
    result = df.copy()

    if category:
        result = result[result['category'].str.lower() == category.lower()]

    if color:
        result = result[result['color'].str.lower() == color.lower()]

    if price_limit:
        result = result[result['price'] <= price_limit]

    if result.empty:
        return {
            "type": "text",
            "message": f"I couldn't find matching products{f' in {color}' if color else ''}{f' under ${price_limit}' if price_limit else ''}{f' in {category}' if category else ''}. Try changing your filters or check for typos! 😊",
            "data": None
        }

    # -----------------------------
    # RECOMMENDATION LOGIC - SIMPLIFIED COLUMNS
    # -----------------------------
    if intent == "cheap":
        result = result[result['price'] > 0].sort_values(by='price').head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are {limit} budget-friendly products{f' in {color}' if color else ''} 💰",
            "data": result[['product_name','category','price','color','popularity_index']]
        }

    elif intent == "best":
        result = result[result['popularity_index'] > 0].sort_values(by='popularity_index', ascending=False).head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are the top {limit} highest-rated products{f' in {color}' if color else ''} ⭐",
            "data": result[['product_name','category','popularity_index','price','color']]
        }

    elif intent == "discount":
        result = result[result['discount'] > 0].sort_values(by='discount', ascending=False).head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are the top {limit} discounted products{f' in {color}' if color else ''} 🔥",
            "data": result[['product_name','category','discount','price','color']]
        }

    else:
        result = result[result['popularity_index'] > 0].sort_values(by='popularity_index', ascending=False).head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are {limit} recommended products{f' in {color}' if color else ''} 👍",
            "data": result[['product_name','category','price','popularity_index','color']]
        }

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            
            if isinstance(content, dict):
                st.write(content["message"])
                
                data_value = content["data"]
                response_type = content.get("type", "")
                
                if response_type == "categories":
                    display_categories()
                elif data_value is not None:
                    if isinstance(data_value, str) and data_value == "SHOW_EXAMPLES":
                        show_examples()
                    elif isinstance(data_value, pd.DataFrame):
                        display_products(data_value, label="Top Recommendations")
            else:
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
            st.write(response["message"])
            
            data_value = response["data"]
            response_type = response.get("type", "")
            
            if response_type == "categories":
                display_categories()
            elif data_value is not None:
                if isinstance(data_value, str) and data_value == "SHOW_EXAMPLES":
                    show_examples()
                elif isinstance(data_value, pd.DataFrame):
                    display_products(data_value, label="Top Recommendations")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
        else:
            st.write(str(response))
            st.session_state.messages.append({
                "role": "assistant",
                "content": str(response)
            })
    
    st.rerun()
    
