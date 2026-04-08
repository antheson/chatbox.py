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
# SHOW EXAMPLE QUESTIONS
# -----------------------------
def show_examples():
    st.markdown("""
### 💡 You can try asking:
- cheap shoes under 100  
- best products  
- shoes above 100  
- clothing under 50  
- shoes between 50 and 150  
- most expensive shoes  
- show me black shoes  
- recommend something  
- show all categories
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
                if 'review_count' in row and row['review_count'] > 0:
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
# EXTRACT PRICE RANGE FROM QUERY - SIMPLIFIED FIXED VERSION
# -----------------------------
def extract_price_range(text):
    """Extract min and max price from user query"""
    text = text.lower()
    min_price = None
    max_price = None
    
    # Look for "above X" or "over X" or "more than X"
    above_match = re.search(r'(?:above|over|more than|greater than)\s+(\d+)', text)
    if above_match:
        min_price = float(above_match.group(1))
        return min_price, max_price
    
    # Look for "under X" or "below X" or "less than X"
    under_match = re.search(r'(?:under|below|less than|cheaper than)\s+(\d+)', text)
    if under_match:
        max_price = float(under_match.group(1))
        return min_price, max_price
    
    # Look for "X-Y" or "X - Y" or "X to Y" or "between X and Y"
    range_match = re.search(r'(\d+)\s*[-–—to]\s*(\d+)', text)
    if range_match:
        min_price = float(range_match.group(1))
        max_price = float(range_match.group(2))
        return min_price, max_price
    
    # Look for "between X and Y"
    between_match = re.search(r'between\s+(\d+)\s+and\s+(\d+)', text)
    if between_match:
        min_price = float(between_match.group(1))
        max_price = float(between_match.group(2))
        return min_price, max_price
    
    # Look for single number (could be max or min)
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        num = float(numbers[0])
        # If "above" or "over" is in text, it's min price
        if 'above' in text or 'over' in text:
            min_price = num
        # If "under" or "below" is in text, it's max price  
        elif 'under' in text or 'below' in text:
            max_price = num
        # If it's a simple query like "shoes 100", treat as max price
        elif len(numbers) == 1 and not any(word in text for word in ['cheap', 'best', 'expensive']):
            max_price = num
    
    return min_price, max_price

# -----------------------------
# CHECK IF QUERY IS VALID
# -----------------------------
def is_valid_query(user_input):
    """Check if the user query is valid for product search"""
    valid_keywords = [
        'cheap', 'best', 'expensive', 'recommend', 'show', 'find', 'get', 'give',
        'shoes', 'clothing', 'accessories', 'originals', 'soccer', 'running',
        'black', 'white', 'blue', 'red', 'pink', 'green', 'purple', 'grey', 'yellow',
        'under', 'below', 'less', 'than', 'above', 'over', 'more', 'between', 'price', 
        'budget', 'affordable', 'top', 'rated', 'popular', 'highest', 'premium', 'luxury'
    ]
    
    # Also check for category names
    categories = [cat.lower() for cat in df['category'].dropna().unique()]
    valid_keywords.extend(categories)
    
    text = user_input.lower()
    return any(keyword in text for keyword in valid_keywords)

# -----------------------------
# RESPONSE GENERATION
# -----------------------------
def get_response(user_input):
    # Apply typo correction
    categories_list = [cat.lower() for cat in df['category'].dropna().unique()]
    corrected_input = correct_category_typo(user_input, categories_list)
    corrected_input = correct_intent_typo(corrected_input)
    text = corrected_input.lower()

    # Predict intent
    try:
        intent = predict_intent(corrected_input)
    except:
        intent = "unknown"
    
    # Check for category intent
    if any(phrase in text for phrase in ["show categories", "what categories", "list categories", "all categories", "available categories"]):
        intent = "categories"

    # Greeting
    if intent == "greeting":
        return {
            "type": "text",
            "message": "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, color, or rating!\n\nTry:\n- cheap shoes under 100\n- shoes above 100\n- clothing under 50\n- shoes between 50 and 150\n- most expensive shoes\n- best clothing",
            "data": "SHOW_EXAMPLES"
        }

    # Thank you responses
    if intent == "thanks":
        thank_messages = [
            "You're very welcome! 😊 Happy shopping! Is there anything else I can help you with?",
            "My pleasure! 🛍️ Let me know if you need more recommendations!",
            "Anytime! 🙌 Feel free to ask if you want to explore more products!",
            "Glad I could help! 🎯 What would you like to look for next?",
            "You got it! 👍 Ready to find your next favorite Adidas item?"
        ]
        import random
        return {
            "type": "text",
            "message": random.choice(thank_messages),
            "data": None
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
    
    # Check if user specified a number
    for word in text.split():
        if word.isdigit():
            requested = int(word)
            if requested < 5:
                limit = requested
            else:
                limit = 5

    # Extract category
    category = None
    for cat in df['category'].dropna().unique():
        if cat.lower() in text:
            category = cat
            break

    # Extract color
    color = None
    colors_list = df['color'].dropna().unique()
    for col in colors_list:
        if col.lower() in text:
            color = col
            break

    # Extract price range (min and max)
    min_price, max_price = extract_price_range(text)

    # -----------------------------
    # INTENT DETECTION
    # -----------------------------
    if "expensive" in text or "premium" in text or "luxury" in text or "most expensive" in text:
        intent = "expensive"
    elif "cheap" in text or "budget" in text or "affordable" in text:
        intent = "cheap"
    elif "best" in text or "top" in text or "popular" in text:
        intent = "best"
    elif "recommend" in text or "show me" in text or "give me" in text:
        intent = "recommend"
    
    # If price range exists without specific intent, treat as price filter
    if (min_price or max_price) and intent not in ["cheap", "expensive"]:
        intent = "price_range"

    # -----------------------------
    # FILTER DATA
    # -----------------------------
    result = df.copy()

    if category:
        result = result[result['category'].str.lower() == category.lower()]

    if color:
        result = result[result['color'].str.lower() == color.lower()]

    if min_price:
        result = result[result['price'] >= min_price]
    
    if max_price:
        result = result[result['price'] <= max_price]

    if result.empty:
        # Build helpful message
        if min_price and max_price:
            price_msg = f"between ${min_price:.0f} and ${max_price:.0f}"
        elif min_price:
            price_msg = f"above ${min_price:.0f}"
        elif max_price:
            price_msg = f"under ${max_price:.0f}"
        else:
            price_msg = ""
        
        return {
            "type": "text",
            "message": f"I couldn't find matching products{f' in {color}' if color else ''} {price_msg}{f' in {category}' if category else ''}. Try changing your filters! 😊",
            "data": None
        }

    # -----------------------------
    # RECOMMENDATION LOGIC
    # -----------------------------
    if intent == "expensive":
        result = result[result['price'] > 0].sort_values(by='price', ascending=False).head(limit)
        price_msg = f"above ${min_price:.0f}" if min_price else ""
        return {
            "type": "dataframe",
            "message": f"Here are the {len(result)} most expensive products{f' in {color}' if color else ''} {price_msg}{f' in {category}' if category else ''} 💎",
            "data": result[['product_name','category','price','color','popularity_index']]
        }
    
    elif intent == "cheap":
        result = result[result['price'] > 0].sort_values(by='price').head(limit)
        price_msg = f"under ${max_price:.0f}" if max_price else ""
        return {
            "type": "dataframe",
            "message": f"Here are {len(result)} budget-friendly products{f' in {color}' if color else ''} {price_msg} 💰",
            "data": result[['product_name','category','price','color','popularity_index']]
        }

    elif intent == "best":
        result = result[result['popularity_index'] > 0].sort_values(by='popularity_index', ascending=False).head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are the top {len(result)} highest-rated products{f' in {color}' if color else ''} ⭐",
            "data": result[['product_name','category','popularity_index','price','color']]
        }

    elif intent == "price_range":
        if min_price and max_price:
            result = result[result['price'] > 0].sort_values(by='price').head(limit)
            return {
                "type": "dataframe",
                "message": f"Here are {len(result)} products between ${min_price:.0f} and ${max_price:.0f}{f' in {color}' if color else ''}{f' in {category}' if category else ''} 💰",
                "data": result[['product_name','category','price','color','popularity_index']]
            }
        elif max_price:
            result = result[result['price'] > 0].sort_values(by='price').head(limit)
            return {
                "type": "dataframe",
                "message": f"Here are {len(result)} products under ${max_price:.0f}{f' in {color}' if color else ''}{f' in {category}' if category else ''} 💰",
                "data": result[['product_name','category','price','color','popularity_index']]
            }
        elif min_price:
            result = result[result['price'] > 0].sort_values(by='price').head(limit)
            return {
                "type": "dataframe",
                "message": f"Here are {len(result)} products above ${min_price:.0f}{f' in {color}' if color else ''}{f' in {category}' if category else ''} 💰",
                "data": result[['product_name','category','price','color','popularity_index']]
            }

    else:
        result = result[result['popularity_index'] > 0].sort_values(by='popularity_index', ascending=False).head(limit)
        return {
            "type": "dataframe",
            "message": f"Here are {len(result)} recommended products{f' in {color}' if color else ''} 👍",
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

# Fixed Clear Chat button at bottom
st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
if st.button("🗑️ Clear Chat", key="clear_chat_bottom"):
    st.session_state.messages = []
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
