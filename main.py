import pandas as pd
import streamlit as st
import re

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")

st.title("🛍️ ShopAssist AI - Adidas Recommendation Chatbot")

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("adidas_usa.csv")
    # Rename columns
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Rename specific columns
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
    
    # Handle price
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'original_price' in df.columns:
        df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
    
    # Fill missing prices
    if 'price' in df.columns and 'original_price' in df.columns:
        df['price'] = df['price'].fillna(df['original_price'])
    
    df['price'] = df['price'].fillna(0)
    
    # Fill NaN values
    df['product_name'] = df['product_name'].fillna('Unknown Product')
    df['category'] = df['category'].fillna('Uncategorized')
    df['color'] = df['color'].fillna('Multiple Colors')
    
    # Remove uncategorized
    df = df[df['category'] != 'Uncategorized']
    
    return df

df = load_data()

# Get unique values for filters
ALL_CATEGORIES = sorted(df['category'].unique())
ALL_COLORS = sorted(df['color'].unique())

# -----------------------------
# PARSE USER QUERY
# -----------------------------
def parse_query(user_input):
    """Parse user input to extract filters"""
    text = user_input.lower()
    
    # Default values
    category = None
    color = None
    min_price = None
    max_price = None
    intent = "recommend"  # default
    
    # Check for specific intents
    if "cheap" in text or "budget" in text or "affordable" in text:
        intent = "cheap"
    elif "expensive" in text or "premium" in text or "luxury" in text:
        intent = "expensive"
    elif "best" in text or "top" in text or "popular" in text or "highest rated" in text:
        intent = "best"
    
    # Extract category
    for cat in ALL_CATEGORIES:
        if cat.lower() in text:
            category = cat
            break
    
    # Extract color - case insensitive matching
    for col in ALL_COLORS:
        if col.lower() in text:
            color = col
            break
    
    # Extract price conditions
    # Pattern: under X, below X, less than X
    under_match = re.search(r'(?:under|below|less than|cheaper than)\s*\$?\s*(\d+(?:\.\d+)?)', text)
    if under_match:
        max_price = float(under_match.group(1))
    
    # Pattern: above X, over X, more than X
    above_match = re.search(r'(?:above|over|more than|greater than)\s*\$?\s*(\d+(?:\.\d+)?)', text)
    if above_match:
        min_price = float(above_match.group(1))
    
    # Pattern: between X and Y
    between_match = re.search(r'between\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*\$?\s*(\d+(?:\.\d+)?)', text)
    if between_match:
        min_price = float(between_match.group(1))
        max_price = float(between_match.group(2))
    
    # Pattern: X to Y
    to_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)', text)
    if to_match and not between_match:
        min_price = float(to_match.group(1))
        max_price = float(to_match.group(2))
    
    # If just a number like "shoes 100" - treat as max price
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    if numbers and not min_price and not max_price:
        num = float(numbers[0])
        # If it's a single number and not part of a range
        if len(numbers) == 1:
            max_price = num
    
    return {
        'category': category,
        'color': color,
        'min_price': min_price,
        'max_price': max_price,
        'intent': intent,
        'raw_text': text
    }

# -----------------------------
# FILTER PRODUCTS
# -----------------------------
def filter_products(df, filters):
    """Apply filters to dataframe"""
    result = df.copy()
    
    if filters['category']:
        result = result[result['category'] == filters['category']]
    
    if filters['color']:
        result = result[result['color'] == filters['color']]
    
    if filters['min_price']:
        result = result[result['price'] >= filters['min_price']]
    
    if filters['max_price']:
        result = result[result['price'] <= filters['max_price']]
    
    return result

# -----------------------------
# GET RESPONSE BASED ON INTENT
# -----------------------------
def get_products_by_intent(filtered_df, intent, limit=5):
    """Sort products based on intent"""
    if filtered_df.empty:
        return filtered_df
    
    if intent == "cheap":
        return filtered_df[filtered_df['price'] > 0].sort_values('price').head(limit)
    elif intent == "expensive":
        return filtered_df[filtered_df['price'] > 0].sort_values('price', ascending=False).head(limit)
    elif intent == "best":
        return filtered_df[filtered_df['popularity_index'] > 0].sort_values('popularity_index', ascending=False).head(limit)
    else:  # recommend
        return filtered_df[filtered_df['popularity_index'] > 0].sort_values('popularity_index', ascending=False).head(limit)

# -----------------------------
# GENERATE RESPONSE MESSAGE
# -----------------------------
def generate_message(filters, result_count, intent):
    """Generate a friendly response message"""
    parts = []
    
    if intent == "cheap":
        parts.append("💰 Here are some budget-friendly")
    elif intent == "expensive":
        parts.append("💎 Here are the most expensive")
    elif intent == "best":
        parts.append("⭐ Here are the highest-rated")
    else:
        parts.append("👍 Here are some recommended")
    
    # Add color
    if filters['color']:
        parts.append(f"{filters['color'].lower()}")
    
    # Add category
    if filters['category']:
        parts.append(f"{filters['category'].lower()}")
    else:
        parts.append("products")
    
    # Add price info
    if filters['min_price'] and filters['max_price']:
        parts.append(f"between ${filters['min_price']:.0f} and ${filters['max_price']:.0f}")
    elif filters['max_price']:
        parts.append(f"under ${filters['max_price']:.0f}")
    elif filters['min_price']:
        parts.append(f"above ${filters['min_price']:.0f}")
    
    message = " ".join(parts)
    return f"{message} 🛍️"

# -----------------------------
# DISPLAY PRODUCTS
# -----------------------------
def display_products(df_result):
    if df_result.empty:
        st.warning("No products found matching your criteria. Try different filters! 😊")
        return
    
    for i, (_, row) in enumerate(df_result.iterrows(), start=1):
        with st.container():
            st.markdown(f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                background-color: #fafafa;
            ">
                <h4 style="margin-top: 0;">#{i} {row.get('product_name', 'Unknown')}</h4>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"📂 **Category:** {row.get('category', 'N/A')}")
            with col2:
                st.write(f"🎨 **Color:** {row.get('color', 'N/A')}")
            with col3:
                if row['price'] > 0:
                    st.write(f"💰 **Price:** ${row['price']:.2f}")
            with col4:
                if row.get('popularity_index', 0) > 0:
                    rating = row['popularity_index']
                    stars = "⭐" * int(round(rating))
                    st.write(f"⭐ **Rating:** {stars} ({rating}/5)")
            
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# SHOW HELP
# -----------------------------
def show_help():
    st.markdown("""
    ### 💡 Try these example queries:
    
    **By Price:**
    - `shoes under 100`
    - `clothing under 50`
    - `shoes above 100`
    - `shoes between 50 and 150`
    
    **By Color:**
    - `show me black shoes`
    - `white shoes`
    - `blue clothing`
    - `pink accessories`
    
    **By Rating:**
    - `best shoes`
    - `top rated clothing`
    - `most popular products`
    
    **Combined:**
    - `cheap black shoes under 80`
    - `best white running shoes`
    - `blue clothing under 50`
    
    **Other:**
    - `show all categories` - See what categories are available
    - `most expensive shoes` - Premium products
    - `recommend something` - Get suggestions
    """)

# -----------------------------
# SHOW CATEGORIES
# -----------------------------
def show_categories():
    st.subheader("📚 Available Categories")
    
    cols = st.columns(3)
    for idx, category in enumerate(ALL_CATEGORIES):
        with cols[idx % 3]:
            product_count = len(df[df['category'] == category])
            st.write(f"• **{category}** ({product_count} products)")
    
    # Show sample of available colors
    with st.expander("🎨 Available Colors"):
        st.write(", ".join(sorted(ALL_COLORS)))

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    # Add welcome message
    st.session_state.messages = [{
        "role": "assistant",
        "content": {
            "type": "welcome",
            "message": "👋 Hi! I'm your Adidas shopping assistant. Try asking me:\n\n• shoes under 100\n• clothing under 50\n• show me black shoes\n• best shoes\n• or type 'help' for more examples!"
        }
    }]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        
        if isinstance(content, dict):
            st.write(content["message"])
            
            if content.get("type") == "categories":
                show_categories()
            elif content.get("type") == "help":
                show_help()
            elif content.get("type") == "products" and "data" in content:
                display_products(content["data"])
        else:
            st.write(content)

# Clear chat button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": {
                "type": "welcome",
                "message": "👋 Chat cleared! Try asking me:\n\n• shoes under 100\n• clothing under 50\n• show me black shoes"
            }
        }]
        st.rerun()

# User input
user_input = st.chat_input("Ask for product recommendations...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # Process the query
    text_lower = user_input.lower()
    
    # Check for special commands
    if text_lower in ["help", "what can you do", "how to use", "commands"]:
        response = {
            "type": "help",
            "message": "Here's what I can help you with! 😊"
        }
    
    elif "categories" in text_lower or "what categories" in text_lower:
        response = {
            "type": "categories",
            "message": "Here are all the product categories available!"
        }
    
    else:
        # Parse the query
        filters = parse_query(user_input)
        
        # Filter products
        filtered_df = filter_products(df, filters)
        
        # Sort based on intent
        result_df = get_products_by_intent(filtered_df, filters['intent'])
        
        if result_df.empty:
            response = {
                "type": "text",
                "message": f"😕 I couldn't find any products matching your criteria. Try something like:\n\n• shoes under 100\n• clothing under 50\n• show me black shoes\n• best shoes"
            }
        else:
            # Generate response message
            message = generate_message(filters, len(result_df), filters['intent'])
            
            response = {
                "type": "products",
                "message": message,
                "data": result_df
            }
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.write(response["message"])
        
        if response.get("type") == "categories":
            show_categories()
        elif response.get("type") == "help":
            show_help()
        elif response.get("type") == "products" and "data" in response:
            display_products(response["data"])
    
    st.rerun()
