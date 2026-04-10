import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from difflib import get_close_matches
import re

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")

# Custom CSS to fix button at bottom
st.markdown("""
<style>
    /* Fix clear chat button at bottom */
    .fixed-bottom {
        position: fixed;
        bottom: 80px;
        right: 20px;
        z-index: 999;
    }
    
    /* Adjust main content to not overlap with button */
    .main > div {
        padding-bottom: 100px;
    }
    
    /* Style the button */
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 20px;
        padding: 8px 20px;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .stButton button:hover {
        background-color: #ff0000;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛍️ ShopAssist AI - Adidas Recommendation Chatbot")

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
    
    # Fill NaN values
    df['product_name'] = df['product_name'].fillna('Unknown Product')
    df['category'] = df['category'].fillna('Uncategorized')
    df['color'] = df['color'].fillna('Multiple Colors')
    
    # REMOVE UNCATEGORIZED PRODUCTS
    df = df[df['category'] != 'Uncategorized']
    
    return df

df = load_data()

# Get all categories and colors for matching
ALL_CATEGORIES = df['category'].dropna().unique().tolist()
ALL_COLORS = df['color'].dropna().unique().tolist()

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
        'expensive': ['expensive', 'expensiv', 'expen', 'costly', 'premium', 'luxury'],
        'categories': ['categories', 'catagories', 'categries', 'catgories', 'categorys'],
        'recommend': ['recommend', 'recomend', 'reccomend', 'rekomend', 'recommanded'],
        'thank': ['thank', 'thanks', 'thx', 'thankyou', 'thank u', 'tq', 'ty']
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
    ("expensive products", "expensive"),
    ("most expensive", "expensive"),
    ("premium products", "expensive"),
    ("show categories", "categories"),
    ("what categories", "categories"),
    ("list categories", "categories"),
    ("thank", "thanks"),
    ("thanks", "thanks"),
    ("thank you", "thanks")
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
# SHOW EXAMPLE QUESTIONS (ONLY WHEN HELP IS REQUESTED)
# -----------------------------
def show_examples():
    st.markdown("""
### 💡 You can try asking:
- shoes under 100  
- clothing under 50  
- white clothing  
- black shoes  
- blue shoes under 80  
- best shoes  
- shoes between 50 and 150  
- most expensive shoes  
- show all categories
""")

# -----------------------------
# DISPLAY PRODUCTS (CLEAN MINIMAL CARDS)
# -----------------------------
def display_products(df_result, label="Recommended Products"):
    if df_result.empty:
        st.warning("No products found.")
        return

    st.subheader(f"🏆 {label}")

    for i, (_, row) in enumerate(df_result.iterrows(), start=1):
        name      = row.get('product_name', 'Unknown')
        category  = row.get('category', 'N/A')
        color     = row.get('color', 'N/A')
        price     = row.get('price', 0)
        price_str = f"${price:.2f}" if price and price > 0 else "N/A"

        st.markdown(f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 10px;
            background-color: #fafafa;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div style="flex: 1; min-width: 0;">
                <p style="margin: 0 0 4px 0; font-weight: 600; font-size: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    #{i} &nbsp; {name}
                </p>
                <p style="margin: 0; color: #888; font-size: 13px;">
                    📂 {category} &nbsp;·&nbsp; 🎨 {color}
                </p>
            </div>
            <div style="margin-left: 20px; text-align: right; flex-shrink: 0;">
                <p style="margin: 0; font-size: 18px; font-weight: 700; color: #2e7d32;">{price_str}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
# EXTRACT ALL FILTERS FROM QUERY
# -----------------------------
def extract_filters(text):
    text_lower = text.lower()

    filters = {
        'category': None,
        'color': None,
        'min_price': None,
        'max_price': None,
        'intent': 'recommend'
    }

    # -----------------------------
    # PRICE DETECTION (FINAL FIX)
    # -----------------------------
    import re
    
    # Pattern: between X and Y OR X-Y
    between_match = re.search(r'(?:between\s+)?(\d+)\s*(?:and|-)\s*(\d+)', text_lower)
    
    if between_match:
        filters['min_price'] = float(between_match.group(1))
        filters['max_price'] = float(between_match.group(2))
    
    else:
        # Only run these if BETWEEN not detected
    
        under_match = re.search(r'(under|below|less than)\s+(\d+)', text_lower)
        if under_match:
            filters['max_price'] = float(under_match.group(2))
    
        above_match = re.search(r'(above|over|more than)\s+(\d+)', text_lower)
        if above_match:
            filters['min_price'] = float(above_match.group(2))
    
        to_match = re.search(r'(\d+)\s+to\s+(\d+)', text_lower)
        if to_match:
            filters['min_price'] = float(to_match.group(1))
            filters['max_price'] = float(to_match.group(2))
    
        # Fallback (ONLY if nothing else detected)
        if filters['min_price'] is None and filters['max_price'] is None:
            numbers = re.findall(r'\b(\d+)\b', text_lower)
            if len(numbers) == 1:
                filters['max_price'] = float(numbers[0])
    
    
    # -----------------------------
    # INTENT DETECTION (MOVE OUTSIDE)
    # -----------------------------
    if 'cheap' in text_lower:
        filters['intent'] = 'cheap'
    elif 'expensive' in text_lower or 'premium' in text_lower:
        filters['intent'] = 'expensive'
    elif 'best' in text_lower or 'top' in text_lower:
        filters['intent'] = 'best'
    elif filters['min_price'] or filters['max_price']:
        filters['intent'] = 'price_range'
    
    
    return filters


    # -----------------------------
    # CATEGORY DETECTION (IMPROVED)
    # -----------------------------
    for cat in ALL_CATEGORIES:
        if cat.lower() in text_lower:
            filters['category'] = cat
            break

    if not filters['category']:
        if any(word in text_lower for word in ['shoe', 'sneaker', 'trainer']):
            filters['category'] = 'shoes'
        elif any(word in text_lower for word in ['cloth', 'shirt', 'pants', 'jacket', 'hoodie']):
            filters['category'] = 'clothing'
        elif any(word in text_lower for word in ['bag', 'sock', 'hat', 'accessor']):
            filters['category'] = 'accessories'

    # -----------------------------
    # COLOR DETECTION (IMPROVED)
    # -----------------------------
    color_keywords = [
        'black','white','blue','red','green','yellow',
        'pink','purple','grey','gray','beige','gold'
    ]

    for color in color_keywords:
        if color in text_lower:
            filters['color'] = color
            break


# -----------------------------
# RESPONSE GENERATION
# -----------------------------
def get_response(user_input):
    # Apply typo correction
    categories_list = [cat.lower() for cat in df['category'].dropna().unique()]
    corrected_input = correct_category_typo(user_input, categories_list)
    corrected_input = correct_intent_typo(corrected_input)
    
    # -----------------------------
    # "MORE RESULTS" DETECTION
    # -----------------------------
    more_keywords = ['more', 'show more', 'next', 'more results', 'see more', 'give more', 'load more']
    is_more_request = any(kw in corrected_input.lower() for kw in more_keywords)

    if is_more_request and st.session_state.last_filters is not None:
        st.session_state.result_offset += 5
        filters = st.session_state.last_filters
    else:
        # Extract filters FIRST
        filters = extract_filters(user_input)
        st.session_state.last_filters = filters
        st.session_state.result_offset = 0

    # Only use ML if no filters detected
    model_intent = "unknown"
    if not filters['category'] and not filters['color'] and not filters['min_price'] and not filters['max_price']:
        try:
            model_intent = predict_intent(corrected_input)
        except:
            model_intent = "unknown"
    
    # Check for special intents
    text_lower = corrected_input.lower()
    
    if any(phrase in text_lower for phrase in ["show categories", "what categories", "list categories", "all categories", "available categories"]):
        return {
            "type": "categories",
            "message": "Here are all the product categories available in our store! 🛍️",
            "data": None
        }
    
    if model_intent == "greeting":
        return {
            "type": "text",
            "message": "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, color, or rating!",
            "data": None
        }
    
    # If user clearly asking for products → skip greeting
    if not filters['category'] and not filters['color'] and not filters['min_price'] and not filters['max_price']:
    
        if model_intent == "greeting":
            return {
                "type": "text",
                "message": "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, color, or rating!",
                "data": None
            }
    
        if model_intent == "thanks":
            import random
            return {
                "type": "text",
                "message": random.choice([
                    "You're very welcome! 😊",
                    "My pleasure! 🛍️",
                    "Anytime! 🙌"
                ]),
                "data": None
            }
    
        if model_intent == "help":
            return {
                "type": "help",
                "message": "Here are some things you can ask me 😊",
                "data": None
            }
    
    # Set limit and offset
    limit = 5
    offset = st.session_state.result_offset
    
    # Apply filters to dataframe
    result = df.copy()
    
    # -----------------------------
    # APPLY FILTERS (FIXED)
    # -----------------------------
    if filters['category']:
        result = result[result['category'].str.contains(filters['category'], case=False, na=False)]
    
    if filters['color']:
        result = result[result['color'].str.contains(filters['color'], case=False, na=False)]
    
    if filters['min_price']:
        result = result[result['price'] >= filters['min_price']]
    
    if filters['max_price']:
        result = result[result['price'] <= filters['max_price']]
    
    # Check if we have results
    if result.empty:
        # Build helpful message
        conditions = []
        if filters['color']:
            conditions.append(f"color '{filters['color']}'")
        if filters['category']:
            conditions.append(f"category '{filters['category']}'")
        if filters['min_price'] and filters['max_price']:
            conditions.append(f"price between ${filters['min_price']:.0f} and ${filters['max_price']:.0f}")
        elif filters['max_price']:
            conditions.append(f"price under ${filters['max_price']:.0f}")
        elif filters['min_price']:
            conditions.append(f"price above ${filters['min_price']:.0f}")
        
        if conditions:
            msg = f"I couldn't find products with {', '.join(conditions)}. Try different filters! 😊"
        else:
            msg = "I couldn't find matching products. Try asking for 'shoes under 100' or 'white clothing'! 😊"
        
        return {
            "type": "text",
            "message": msg,
            "data": None
        }
    
    # Sort based on intent
    if filters['intent'] == 'expensive':
        sorted_result = result[result['price'] > 0].sort_values('price', ascending=False)
        msg = f"Here are the most expensive products"
    elif filters['intent'] == 'cheap':
        sorted_result = result[result['price'] > 0].sort_values('price')
        msg = f"Here are budget-friendly products"
    elif filters['intent'] == 'best':
        sorted_result = result[result['popularity_index'] > 0].sort_values('popularity_index', ascending=False)
        msg = f"Here are the highest-rated products"
    elif filters['intent'] == 'price_range':
        sorted_result = result[result['price'] > 0].sort_values('price')
        if filters['min_price'] and filters['max_price']:
            msg = f"Here are products between ${filters['min_price']:.0f} and ${filters['max_price']:.0f}"
        elif filters['max_price']:
            msg = f"Here are products under ${filters['max_price']:.0f}"
        elif filters['min_price']:
            msg = f"Here are products above ${filters['min_price']:.0f}"
        else:
            msg = f"Here are recommended products"
    else:
        sorted_result = result[result['popularity_index'] > 0].sort_values('popularity_index', ascending=False)
        msg = f"Here are recommended products"

    # Slice with offset
    page_result = sorted_result.iloc[offset:offset + limit]

    if page_result.empty:
        st.session_state.result_offset = max(0, offset - limit)
        return {
            "type": "text",
            "message": "No more results to show! Try a different search 😊",
            "data": None
        }

    total_shown = offset + len(page_result)
    total_available = len(sorted_result)
    msg += f" (showing {offset + 1}–{total_shown} of {total_available})"
    
    # Add color to message
    if filters['color']:
        msg += f" in {filters['color']}"
    
    # Add category to message
    if filters['category']:
        msg += f" in {filters['category']}"
    
    if total_shown < total_available:
        msg += " 🛍️ — say **'more'** to see more!"
    else:
        msg += " 🛍️ — that's all the results!"
    
    return {
        "type": "dataframe",
        "message": msg,
        "data": page_result[['product_name', 'category', 'price', 'color', 'popularity_index']]
    }

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_filters" not in st.session_state:
    st.session_state.last_filters = None
if "result_offset" not in st.session_state:
    st.session_state.result_offset = 0

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
                elif response_type == "help":
                    show_examples()
                elif data_value is not None:
                    if isinstance(data_value, str) and data_value == "SHOW_EXAMPLES":
                        show_examples()
                    elif isinstance(data_value, pd.DataFrame):
                        display_products(data_value, label="Top Recommendations")
            else:
                st.write(content)

# Fixed Clear Chat button at bottom
st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
if st.button("🗑️ Clear Chat", key="clear_chat_bottom"):
    st.session_state.messages = []
    st.session_state.last_filters = None
    st.session_state.result_offset = 0
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

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
            elif response_type == "help":
                show_examples()
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
