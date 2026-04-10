import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from difflib import get_close_matches
import re

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️")  # icon kept as emoji (external URLs not supported in page_icon)

# Custom CSS
st.markdown("""
<style>
    /* ── Page background ── */
    .stApp { background-color: #f7f8fc; }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
    section[data-testid="stSidebar"] h2 { color: #ffffff !important; letter-spacing: 0.5px; }
    section[data-testid="stSidebar"] hr { border-color: #2e3a5c !important; }

    /* Sidebar conversation buttons */
    section[data-testid="stSidebar"] .stButton button {
        text-align: left;
        background-color: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px;
        color: #c5cae9 !important;
        font-weight: normal;
        box-shadow: none;
        padding: 6px 12px;
        margin-bottom: 2px;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(255,255,255,0.14) !important;
        color: #ffffff !important;
    }

    /* ── Red View → button only ── */
    .view-btn button {
        background-color: #e53935 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .view-btn button:hover {
        background-color: #b71c1c !important;
        color: white !important;
    }

    /* ── "Got it" welcome button ── */
    .welcome-btn button {
        background: linear-gradient(90deg, #1a237e, #3949ab) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Adidas logo + title
st.markdown(
    "<div style='display:flex;align-items:center;gap:14px;padding:14px 20px;"
    "background:linear-gradient(90deg,#1a237e 0%,#283593 60%,#e53935 100%);"
    "border-radius:14px;margin-bottom:18px;'>"
    "<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Adidas_Logo.svg/2560px-Adidas_Logo.svg.png'"
    " style='height:32px;filter:brightness(0) invert(1);' />"
    "<div>"
    "<div style='font-size:20px;font-weight:700;color:#ffffff;letter-spacing:0.5px;'>ShopAssist AI</div>"
    "<div style='font-size:12px;color:#c5cae9;margin-top:1px;'>Adidas USA · Smart Product Finder</div>"
    "</div>"
    "</div>",
    unsafe_allow_html=True
)

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
        df['original_price'] = df['original_price'].astype(str).str.replace(r'[\$,\s]', '', regex=True)
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

    # Extract gender from breadcrumbs (e.g. "Women/Shoes" -> "Women")
    def extract_gender(breadcrumb):
        if pd.isna(breadcrumb):
            return 'Unisex'
        b = str(breadcrumb).lower()
        if b.startswith('women'):
            return 'Women'
        elif b.startswith('men'):
            return 'Men'
        elif b.startswith('kids'):
            return 'Kids'
        else:
            return 'Unisex'
    df['gender'] = df['breadcrumbs'].apply(extract_gender)

    # Keep first image URL only
    df['image_url'] = df['images'].apply(
        lambda x: str(x).split('~')[0].strip() if pd.notna(x) else ''
    )

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
        'thank': ['thank', 'thanks', 'thx', 'thankyou', 'thank u', 'tq', 'ty'],
        'shoes': ['shoes', 'shoees', 'sheos', 'shoess', 'shos', 'shose', 'suoes', 'suoe'],
        'clothing': ['clothing', 'vlothing', 'cloting', 'clohting', 'clothng', 'clthing',
                     'clothin', 'clotthing', 'clething', 'cloding', 'clohing', 'cloathing'],
        'accessories': ['accessories', 'acessories', 'accesories', 'accessorys', 'acessory',
                        'accessoires', 'accesories', 'accesory'],
        'slides': ['slides', 'slids', 'slids', 'sldies', 'sldes'],
        'sandals': ['sandals', 'sandal', 'sandels', 'sandels', 'sandlas'],
        'running': ['running', 'runing', 'runnin', 'runing', 'runninng'],
        'casual': ['casual', 'cazual', 'casaul', 'casuel', 'causal'],
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
# ✨ SMART EXPLANATION GENERATOR
# -----------------------------
def generate_explanation(row, filters):
    """
    Pick the top 3 most relevant reasons (priority-ordered) and return
    them as short pill labels, e.g. ["✔ In budget", "⭐ 4.8/5", "🔥 15% off"]
    """
    pills = []

    price        = row.get('price', 0)
    orig_price   = row.get('original_price', 0)
    rating       = row.get('popularity_index', 0)
    color        = row.get('color', '')
    gender       = row.get('gender', '')
    availability = row.get('availability', '')
    review_count = row.get('review_count', 0)
    intent       = filters.get('intent', '')
    max_price    = filters.get('max_price')
    min_price    = filters.get('min_price')

    # Priority 1 — direct filter matches (most relevant to the user's query)
    if max_price and price and price <= max_price:
        pills.append(f"✔ In budget")
    elif min_price and max_price and price:
        pills.append(f"✔ In price range")

    if filters.get('color') and color and filters['color'].lower() in color.lower():
        pills.append(f"✔ {color} color")

    if filters.get('gender') and gender == filters['gender']:
        pills.append(f"✔ For {gender}")

    if filters.get('subcategory'):
        pills.append(f"✔ {filters['subcategory'].capitalize()}")

    # Priority 2 — rating (always useful, keep short)
    try:
        r = float(rating)
        if r >= 4.8:
            pills.append(f"⭐ {r}/5 rated")
        elif r >= 4.5:
            pills.append(f"⭐ {r}/5 rated")
    except Exception:
        pass

    # Priority 3 — discount (only if notable)
    try:
        orig = float(orig_price) if orig_price else 0
        cur  = float(price) if price else 0
        if orig > 0 and cur > 0 and orig > cur:
            pct = round((orig - cur) / orig * 100)
            if pct >= 10:
                pills.append(f"🔥 {pct}% off")
    except Exception:
        pass

    # Priority 4 — intent-specific
    if intent == 'cheap' and price:
        pills.append(f"💰 ${price:.0f}")
    elif intent == 'expensive' and price:
        pills.append(f"💎 Premium")

    # Priority 5 — availability (only if nothing else)
    if availability == 'InStock' and len(pills) == 0:
        pills.append("✔ In stock")

    # Fallback
    if not pills:
        try:
            r = float(rating)
            if r > 0:
                pills.append(f"⭐ {r}/5")
        except Exception:
            pass
        pills.append("✔ Matches search")

    return pills[:3]   # cap at 3 pills max


def render_explanation(pills):
    """Render pills as a single subtle line at the bottom of the card."""
    pill_html = "".join(
        f"<span style='background:#e8f5e9;color:#2e7d32;border-radius:12px;"
        f"padding:2px 9px;font-size:11px;margin-right:5px;white-space:nowrap;'>{p}</span>"
        for p in pills
    )
    st.markdown(
        f"<div style='margin-top:3px;margin-bottom:2px;padding-left:2px;'>{pill_html}</div>",
        unsafe_allow_html=True
    )


# -----------------------------
# DISPLAY PRODUCT DETAIL
# -----------------------------
def display_product_detail(row):
    name         = row.get('product_name', 'Unknown')
    category     = row.get('category', 'N/A')
    color        = row.get('color', 'N/A')
    price        = row.get('price', 0)
    orig_price   = row.get('original_price', 0)
    rating       = row.get('popularity_index', 0)
    reviews      = row.get('review_count', 0)
    gender       = row.get('gender', 'Unisex')
    availability = row.get('availability', 'N/A')
    description  = row.get('description', 'No description available.')
    image_url    = row.get('image_url', '')

    price_str   = f"${price:.2f}" if price and price > 0 else "N/A"
    avail_color = "#2e7d32" if availability == "InStock" else "#c62828"
    avail_label = "✅ In Stock" if availability == "InStock" else "❌ Out of Stock"
    rating_str  = (f"{'\u2b50' * int(round(float(rating)))}{'\u2606' * (5 - int(round(float(rating))))} ({rating}/5)") if rating and float(rating) > 0 else "No rating"
    reviews_str = f"{int(reviews):,}" if reviews and float(str(reviews).replace(',','')) > 0 else "No reviews"

    col_img, col_info = st.columns([1, 2])
    with col_img:
        if image_url:
            st.image(image_url, use_container_width=True)
        else:
            st.markdown("🖼️ *No image available*")
    with col_info:
        st.markdown(f"### {name}")
        st.markdown(f"<span style='color:{avail_color}; font-weight:600;'>{avail_label}</span>", unsafe_allow_html=True)
        st.markdown(f"**💰 Price:** {price_str}")
        if orig_price and float(str(orig_price)) > float(str(price)) and float(str(price)) > 0:
            st.markdown(f"<s style='color:#999'>Original: ${float(orig_price):.2f}</s>", unsafe_allow_html=True)
        st.markdown(f"**📂 Category:** {category} &nbsp;·&nbsp; **🎨 Color:** {color}")
        st.markdown(f"**👤 Gender:** {gender} &nbsp;·&nbsp; **⭐ Rating:** {rating_str}")
        st.markdown(f"**📝 Reviews:** {reviews_str}")
        st.markdown("---")
        st.markdown("**📄 Description:**")
        st.markdown(f"_{description}_")

    if st.button("← Back to results", key="back_btn"):
        st.session_state.selected_product = None
        st.rerun()

# -----------------------------
# DISPLAY PRODUCTS (CLICKABLE CARDS) — with Smart Explanation
# -----------------------------
def display_products(df_result, label="Recommended Products", card_key_prefix="card", filters=None):
    if df_result.empty:
        st.warning("No products found.")
        return

    st.markdown(
        f"<div style='background:linear-gradient(90deg,#1a237e,#3949ab);color:white;"
        f"border-radius:8px;padding:8px 16px;margin-bottom:10px;font-weight:600;font-size:16px;'>"
        f"🏆 {label}</div>",
        unsafe_allow_html=True
    )

    for i, (idx, row) in enumerate(df_result.iterrows(), start=1):
        name      = row.get('product_name', 'Unknown')
        category  = row.get('category', 'N/A')
        color     = row.get('color', 'N/A')
        gender    = row.get('gender', '')
        price     = row.get('price', 0)
        price_str = f"${price:.2f}" if price and price > 0 else "N/A"
        gender_badge = (
            f" &nbsp;<span style='background:#e3f2fd;color:#1565c0;"
            f"border-radius:4px;padding:1px 6px;font-size:11px;'>{gender}</span>"
        ) if gender and gender != 'Unisex' else ""

        col_card, col_btn = st.columns([5, 1])
        with col_card:
            pill_line = ""
            if filters is not None:
                pills = generate_explanation(row.to_dict(), filters)
                pill_html = "".join(
                    f"<span style='background:#e8f5e9;color:#2e7d32;border-radius:12px;"
                    f"padding:2px 9px;font-size:11px;margin-right:5px;white-space:nowrap;'>{p}</span>"
                    for p in pills
                )
                pill_line = f"<p style='margin:6px 0 0 0;'>{pill_html}</p>"

            st.markdown(
                f'<div style="border:1px solid #e0e0e0;border-left:4px solid #3949ab;'
                f'border-radius:10px;padding:14px 18px;background:#ffffff;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<p style="margin:0;font-weight:600;font-size:15px;color:#1a237e;">#{i} &nbsp; {name}{gender_badge}</p>'
                f'<span style="font-weight:700;font-size:16px;color:#2e7d32;">{price_str}</span>'
                f'</div>'
                f'<p style="margin:5px 0 0 0;font-size:13px;color:#666;">'
                f'<span style="background:#e8eaf6;color:#3949ab;border-radius:4px;padding:1px 7px;font-size:11px;margin-right:6px;">📂 {category}</span>'
                f'<span style="color:#888;">🎨 {color}</span>'
                f'</p>'
                f'{pill_line}'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_btn:
            st.markdown('<div class="view-btn">', unsafe_allow_html=True)
            if st.button("View →", key=f"{card_key_prefix}_{i}_{idx}"):
                st.session_state.selected_product = row.to_dict()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

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
    # CATEGORY DETECTION
    # -----------------------------
    for cat in ALL_CATEGORIES:
        if cat.lower() in text_lower:
            filters['category'] = cat
            break

    if not filters['category']:
        if any(word in text_lower for word in ['shoe', 'shoes', 'sneaker', 'trainer']):
            filters['category'] = 'Shoes'
        elif any(word in text_lower for word in ['cloth', 'clothing', 'shirt', 'pants', 'jacket', 'hoodie']):
            filters['category'] = 'Clothing'
        elif any(word in text_lower for word in ['bag', 'sock', 'hat', 'accessor']):
            filters['category'] = 'Accessories'

    # -----------------------------
    # COLOR DETECTION
    # -----------------------------
    # Map user color terms to actual dataset color values
    color_alias_map = {
        'black': 'Black',
        'white': 'White',
        'blue': 'Blue',
        'red': 'Red',
        'green': 'Green',
        'yellow': 'Yellow',
        'pink': 'Pink',
        'purple': 'Purple',
        'grey': 'Grey',
        'gray': 'Grey',
        'beige': 'Beige',
        'gold': 'Gold',
        'orange': 'Red',       # closest in dataset
        'brown': 'Beige',      # closest in dataset
        'burgundy': 'Burgundy',
        'multicolor': 'Multicolor',
        'multi color': 'Multicolor',
        'colorful': 'Multicolor',
        'multi': 'Multicolor',
        'rainbow': None,        # not in dataset → trigger no-result message
    }

    for user_color, dataset_color in color_alias_map.items():
        if re.search(rf'\b{re.escape(user_color)}\b', text_lower):
            filters['color'] = dataset_color   # None means "searched but not found"
            filters['color_searched'] = user_color
            break

    # -----------------------------
    # SUBCATEGORY / PRODUCT TYPE DETECTION
    # Based on actual dataset product names
    # -----------------------------

    # Map user terms -> product name keywords to search
    subcategory_map = {
        'running':   {
            'user_terms': ['running', 'run', 'jogging', 'jog'],
            'name_keywords': ['run', 'boost', 'pureboost', 'ultraboost', 'supernova', 'eq21', 'tensor', 'fluidflash']
        },
        'slides': {
            'user_terms': ['slides', 'slide', 'sandal', 'sandals', 'flip flop', 'flip-flop', 'slippers'],
            'name_keywords': ['slide', 'sandal', 'flip', 'mule', 'adilette']
        },
        'soccer': {
            'user_terms': ['soccer', 'football', 'cleat', 'cleats', 'turf'],
            'name_keywords': ['copa', 'predator', 'ghosted', 'cleat', 'turf', 'firm ground']
        },
        'golf': {
            'user_terms': ['golf'],
            'name_keywords': ['golf']
        },
        'climbing': {
            'user_terms': ['climbing', 'climb', 'boulder'],
            'name_keywords': ['climbing', 'hiangle', 'five ten']
        },
        'cycling': {
            'user_terms': ['cycling', 'bike', 'mountain bike', 'mtb'],
            'name_keywords': ['bike', 'kestrel', 'five ten']
        },
        'basketball': {
            'user_terms': ['basketball', 'court'],
            'name_keywords': ['hoops']
        },
        'casual': {
            'user_terms': ['casual', 'lifestyle', 'everyday', 'classic', 'street'],
            'name_keywords': ['superstar', 'samba', 'nizza', 'hamburg', 'ozweego', 'ozelia', 'postmove', 'grand court', 'advantage', 'retrorun', 'sambarose', 'bryony']
        },
        'training': {
            'user_terms': ['training', 'gym', 'workout', 'cross training', 'crossfit'],
            'name_keywords': ['racer', 'swift', 'climacool', 'multix', 'puremotion', 'futurenatural']
        },
        'hiking': {
            'user_terms': ['hiking', 'hike', 'trail', 'outdoor', 'trekking'],
            'name_keywords': ['hike', 'trail', 'outdoor', 'kestrel']
        },
        # ---- CLOTHING SUBCATEGORIES ----
        'hoodie': {
            'user_terms': ['hoodie', 'hoodies', 'hoddie', 'hoody', 'zip up', 'zip-up'],
            'name_keywords': ['hoodie']
        },
        'tee': {
            'user_terms': ['tee', 'tees', 't-shirt', 'tshirt', 't shirt', 'shirt', 'polo'],
            'name_keywords': ['tee', 'polo']
        },
        'sweatshirt': {
            'user_terms': ['sweatshirt', 'sweatshirts', 'sweater', 'crewneck', 'crew neck'],
            'name_keywords': ['sweatshirt']
        },
        'jacket': {
            'user_terms': ['jacket', 'jackets', 'windbreaker', 'wind breaker'],
            'name_keywords': ['jacket', 'windbreaker']
        },
        'pants': {
            'user_terms': ['pants', 'trousers', 'joggers', 'track pants', 'trackpants'],
            'name_keywords': ['pants']
        },
        'shorts': {
            'user_terms': ['shorts', 'short'],
            'name_keywords': ['shorts']
        },
        'tights': {
            'user_terms': ['tights', 'leggings', 'leggins', 'tight'],
            'name_keywords': ['tights']
        },
        'dress': {
            'user_terms': ['dress', 'dresses'],
            'name_keywords': ['dress']
        },
        'tank top': {
            'user_terms': ['tank top', 'tank', 'crop top', 'tanktop'],
            'name_keywords': ['tank top', 'crop top']
        },
        'jersey': {
            'user_terms': ['jersey', 'jerseys', 'kit'],
            'name_keywords': ['jersey']
        },
        'tracksuit': {
            'user_terms': ['tracksuit', 'track suit', 'tracksuits', 'sst'],
            'name_keywords': ['track suit', 'sst set']
        },
        'swimwear': {
            'user_terms': ['swimwear', 'swimsuit', 'swim shorts', 'swimming'],
            'name_keywords': ['swimsuit', 'swim shorts']
        },
    }

    # Clothing subcategories - used to auto-set category to Clothing
    clothing_subcats = {
        'hoodie', 'tee', 'sweatshirt', 'jacket', 'pants', 'shorts',
        'tights', 'dress', 'tank top', 'jersey', 'tracksuit', 'swimwear'
    }

    filters['subcategory'] = None
    filters['subcategory_name_keywords'] = None
    filters['subcategory_category'] = None   # which main category this subcat belongs to

    # Detect ALL matching subcategories (for multi-category combos)
    matched_subcats = []
    for subcat, mapping in subcategory_map.items():
        if any(kw in text_lower for kw in mapping['user_terms']):
            cat = 'Clothing' if subcat in clothing_subcats else 'Shoes'
            matched_subcats.append((subcat, mapping['name_keywords'], cat))

    if len(matched_subcats) == 1:
        filters['subcategory'] = matched_subcats[0][0]
        filters['subcategory_name_keywords'] = matched_subcats[0][1]
        filters['subcategory_category'] = matched_subcats[0][2]
        filters['all_matched_subcats'] = None
    elif len(matched_subcats) > 1:
        # Multi-category combo: store all, handle in get_response
        filters['subcategory'] = matched_subcats[0][0]
        filters['subcategory_name_keywords'] = matched_subcats[0][1]
        filters['subcategory_category'] = matched_subcats[0][2]
        filters['all_matched_subcats'] = matched_subcats   # full list for combined results
    else:
        filters['all_matched_subcats'] = None

    # -----------------------------
    # GENDER DETECTION
    # -----------------------------
    filters['gender'] = None
    if re.search(r"\bwomen'?s?\b|\bfemale\b|\bladies\b|\bgirl\b|\bgirls\b", text_lower):
        filters['gender'] = 'Women'
    elif re.search(r"\bmen'?s?\b|\bmale\b|\bguy\b|\bguys\b|\bboy\b(?!s shoes)\b", text_lower):
        filters['gender'] = 'Men'
    elif re.search(r'\bkids?\b|\bchildren\b|\bjunior\b', text_lower):
        filters['gender'] = 'Kids'

    # -----------------------------
    # INTENT DETECTION
    # -----------------------------
    # Check for negation before expensive ("less expensive", "not expensive", "cheaper than")
    negated_expensive = re.search(r'(less|not|cheaper|more affordable|lower).{0,10}(expensive|costly|premium)', text_lower)
    negated_cheap = re.search(r'(not|less).{0,10}(cheap|budget)', text_lower)

    if negated_expensive:
        filters['intent'] = 'cheap'
    elif negated_cheap:
        filters['intent'] = 'recommend'
    elif 'cheap' in text_lower or 'budget' in text_lower or 'affordable' in text_lower:
        filters['intent'] = 'cheap'
    elif 'expensive' in text_lower or 'premium' in text_lower or 'luxury' in text_lower:
        filters['intent'] = 'expensive'
    elif 'best' in text_lower or 'top' in text_lower or 'highest rated' in text_lower:
        filters['intent'] = 'best'
    elif any(kw in text_lower for kw in ['discount', 'discounted', 'on sale', 'sale', 'deal', 'deals', 'reduced', 'markdown', 'clearance', 'offer', 'promo']):
        filters['intent'] = 'discount'
    elif filters['min_price'] or filters['max_price']:
        filters['intent'] = 'price_range'

    return filters


# -----------------------------
# PRODUCT NAME SEARCH
# -----------------------------
def search_by_product_name(user_input):
    """
    Detect if the user is searching for a specific product by name.
    Returns a DataFrame of matching rows, or None if not a name search.

    Strategy:
      1. Strip known trigger phrases ("find", "show me", "search for", etc.)
      2. Try exact substring match on product names (case-insensitive)
      3. Fall back to fuzzy token match using difflib for typos / partial names
    """
    trigger_phrases = [
        'find ', 'search for ', 'search ', 'show me ', 'look for ',
        'looking for ', 'do you have ', 'do you sell ', 'i want ',
        'i need ', 'get me ', 'show ', 'any '
    ]
    query = user_input.lower().strip()
    for phrase in trigger_phrases:
        if query.startswith(phrase):
            query = query[len(phrase):].strip()

    # Must be at least 4 chars to avoid false positives on short filter words
    if len(query) < 4:
        return None

    # Skip if query looks like a pure filter (color/category/price words only)
    filter_words = {
        'shoes', 'shoe', 'clothing', 'clothes', 'accessories', 'accessory',
        'cheap', 'expensive', 'best', 'top', 'budget', 'premium', 'running',
        'casual', 'slides', 'sandals', 'hoodie', 'jacket', 'pants', 'shorts',
        'black', 'white', 'blue', 'red', 'green', 'pink', 'grey', 'gray',
        'men', 'women', 'kids', 'under', 'above', 'between', 'more'
    }
    query_words = set(query.split())
    if query_words.issubset(filter_words):
        return None

    all_names = df['product_name'].dropna().tolist()
    all_names_lower = [n.lower() for n in all_names]

    # 1. Exact substring match — most reliable
    exact_mask = df['product_name'].str.lower().str.contains(re.escape(query), na=False)
    if exact_mask.any():
        return df[exact_mask]

    # 2. All query words present in name (handles word-order variations)
    words = [w for w in query.split() if len(w) > 2 and w not in filter_words]
    if len(words) >= 2:
        mask = pd.Series([True] * len(df), index=df.index)
        for w in words:
            mask = mask & df['product_name'].str.lower().str.contains(re.escape(w), na=False)
        if mask.any():
            return df[mask]

    # 3. Fuzzy match — catches typos like "ultraboost" → "Ultraboost 21"
    matches = get_close_matches(query, all_names_lower, n=5, cutoff=0.45)
    if matches:
        matched_rows = df[df['product_name'].str.lower().isin(matches)]
        if not matched_rows.empty:
            return matched_rows

    return None


# -----------------------------
# RESPONSE GENERATION
# -----------------------------
def get_response(user_input):
    # Apply typo correction
    categories_list = [cat.lower() for cat in df['category'].dropna().unique()]
    corrected_input = correct_category_typo(user_input, categories_list)
    corrected_input = correct_intent_typo(corrected_input)

    # -----------------------------
    # PRODUCT NAME SEARCH — runs first, before all other logic
    # -----------------------------
    name_results = search_by_product_name(user_input)
    if name_results is not None and not name_results.empty:
        filters = {'category': None, 'color': None, 'min_price': None,
                   'max_price': None, 'intent': 'recommend',
                   'subcategory': None, 'gender': None}
        st.session_state.last_filters = filters
        st.session_state.result_offset = 0
        count = len(name_results)
        label = f'"{user_input.strip()}"'
        msg = (
            f"Found {count} product{'s' if count > 1 else ''} matching {label} 🔍"
            if count > 1 else
            f"Here's the exact product for {label} 🎯"
        )
        cols_needed = ['product_name', 'category', 'price', 'color',
                       'popularity_index', 'review_count', 'gender',
                       'availability', 'description', 'image_url', 'original_price']
        return {
            "type": "dataframe",
            "message": msg,
            "data": name_results[cols_needed].reset_index(drop=True),
            "filters": filters
        }
    
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
    if not filters['category'] and not filters['color'] and not filters['min_price'] and not filters['max_price'] and not filters.get('subcategory'):
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
            "data": None,
            "filters": filters
        }
    
    if model_intent == "greeting":
        return {
            "type": "text",
            "message": "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, color, or rating!",
            "data": None,
            "filters": filters
        }
    
    # If user clearly asking for products → skip greeting
    if not filters['category'] and not filters['color'] and not filters['min_price'] and not filters['max_price'] and not filters.get('subcategory'):
    
        if model_intent == "greeting":
            return {
                "type": "text",
                "message": "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, color, or rating!",
                "data": None,
                "filters": filters
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
                "data": None,
                "filters": filters
            }
    
        if model_intent == "help":
            return {
                "type": "help",
                "message": "Here are some things you can ask me 😊",
                "data": None,
                "filters": filters
            }
    
    # Set limit and offset
    limit = 5
    offset = st.session_state.result_offset
    
    # Apply filters to dataframe
    result = df.copy()
    
    # -----------------------------
    # APPLY FILTERS
    # -----------------------------
    if filters['category']:
        result = result[result['category'].str.contains(filters['category'], case=False, na=False)]

    # Apply color / gender / price BEFORE subcategory splitting
    if filters.get('color_searched') and filters['color'] is None:
        searched = filters['color_searched']
        available = 'black, white, blue, red, green, yellow, pink, purple, grey, beige, gold, burgundy, multicolor'
        return {
            "type": "text",
            "message": f"Sorry, we don't have any '{searched}' products. 😊 Available colors: {available}.",
            "data": None,
            "filters": filters
        }
    if filters['color']:
        result = result[result['color'].str.contains(filters['color'], case=False, na=False)]

    if filters.get('gender'):
        result = result[result['gender'] == filters['gender']]

    if filters['min_price']:
        result = result[result['price'] >= filters['min_price']]

    if filters['max_price']:
        result = result[result['price'] <= filters['max_price']]

    # Subcategory: filter by product name using accurate dataset keywords
    def apply_single_subcat(base_df, name_keywords, subcat_category, main_category_filter):
        """Filter base_df by subcategory name keywords and auto-category."""
        kws = name_keywords
        pattern = '|'.join(re.escape(kw) for kw in kws)
        sub_result = base_df[base_df['product_name'].str.contains(pattern, case=False, na=False)]
        if sub_result.empty:
            sub_result = base_df  # fallback: don't restrict by name
        if subcat_category and not main_category_filter:
            sub_result = sub_result[sub_result['category'].str.contains(subcat_category, case=False, na=False)]
        return sub_result

    all_subcats = filters.get('all_matched_subcats')

    if all_subcats and len(all_subcats) > 1:
        import random
        frames = []
        subcat_names = []
        for (subcat_name, name_kws, subcat_cat) in all_subcats:
            part = apply_single_subcat(result, name_kws, subcat_cat, filters['category'])
            if not part.empty:
                frames.append(part)
                subcat_names.append(subcat_name)

        if frames:
            import random
            limit = 5
            n = len(frames)
            base = limit // n
            remainder = limit % n
            counts = [base + (1 if i < remainder else 0) for i in range(n)]
            parts = []
            for frame, count in zip(frames, counts):
                shuffled = frame.sample(frac=1, random_state=None).reset_index(drop=True)
                parts.append(shuffled.head(count))
            interleaved = []
            max_len = max(len(p) for p in parts)
            for i in range(max_len):
                for p in parts:
                    if i < len(p):
                        interleaved.append(p.iloc[i])
            result = pd.DataFrame(interleaved).drop_duplicates().reset_index(drop=True)
            filters['multi_subcat_note'] = f"Showing a mix of: {' + '.join(subcat_names)} 🎲"
            filters['multi_subcat_limit_applied'] = True

    elif filters.get('subcategory') and filters.get('subcategory_name_keywords'):
        result = apply_single_subcat(result, filters['subcategory_name_keywords'],
                                     filters.get('subcategory_category'), filters['category'])

    # Check if we have results
    if result.empty:
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
        
        if filters.get('subcategory'):
            msg = f"I couldn't find any {filters['subcategory']} products. Try: running shoes, slides, soccer, golf, casual, or training! 😊"
        elif conditions:
            msg = f"I couldn't find products with {', '.join(conditions)}. Try different filters! 😊"
        else:
            msg = "I couldn't find matching products. Try asking for 'shoes under 100' or 'white clothing'! 😊"
        
        return {
            "type": "text",
            "message": msg,
            "data": None,
            "filters": filters
        }
    
    # Sort based on intent
    is_multi_combo = bool(filters.get('all_matched_subcats') and len(filters['all_matched_subcats']) > 1)

    if filters['intent'] == 'expensive':
        sorted_result = result[result['price'] > 0].sort_values('price', ascending=False)
        msg = f"Here are the most expensive products"
    elif filters['intent'] == 'cheap':
        sorted_result = result[result['price'] > 0].sort_values('price')
        msg = f"Here are budget-friendly products"
    elif filters['intent'] == 'best' and not is_multi_combo:
        sorted_result = result[result['popularity_index'] > 0].sort_values('popularity_index', ascending=False)
        msg = f"Here are the highest-rated products"
    elif filters['intent'] == 'discount':
        result['_orig'] = result['original_price']
        result['_disc_amt'] = result['_orig'] - result['price']
        result['_disc_pct'] = ((result['_disc_amt']) / result['_orig'] * 100).round()
        discounted = result[(result['_orig'] > result['price']) & (result['_disc_pct'] >= 5)]
        if discounted.empty:
            discounted = result  # fallback: show all
        sorted_result = discounted.sort_values('_disc_pct', ascending=False)
        sorted_result = sorted_result.drop(columns=['_orig','_disc_amt','_disc_pct'], errors='ignore')
        result = result.drop(columns=['_orig','_disc_amt','_disc_pct'], errors='ignore')
        msg = "Here are products on sale (biggest discounts first)"
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
        if is_multi_combo:
            sorted_result = result
            msg = f"Here's a random mix of products"
        else:
            sorted_result = result[result['popularity_index'] > 0].sort_values('popularity_index', ascending=False)
            msg = f"Here are recommended products"

    # Slice with offset
    if filters.get('multi_subcat_limit_applied') and offset == 0:
        page_result = result
    else:
        page_result = sorted_result.iloc[offset:offset + limit]

    if page_result.empty:
        st.session_state.result_offset = max(0, offset - limit)
        return {
            "type": "text",
            "message": "No more results to show! Try a different search 😊",
            "data": None,
            "filters": filters
        }

    total_shown = offset + len(page_result)
    total_available = len(sorted_result)
    msg += f" (showing {offset + 1}–{total_shown} of {total_available})"
    
    if filters.get('subcategory'):
        msg += f" in {filters['subcategory']}"
    if filters['color']:
        msg += f" · {filters['color']}"
    if filters['category'] and not filters.get('subcategory'):
        msg += f" in {filters['category']}"

    if total_shown < total_available:
        msg += " 🛍️ — say **'more'** to see more!"
    else:
        msg += " 🛍️ — that's all the results!"

    if filters.get('multi_subcat_note'):
        msg += f"\n\n💡 _{filters['multi_subcat_note']}_"
    
    return {
        "type": "dataframe",
        "message": msg,
        "data": page_result[['product_name', 'category', 'price', 'color',
                               'popularity_index', 'review_count', 'gender',
                               'availability', 'description', 'image_url', 'original_price']],
        "filters": filters   # ← pass filters through so display_products can use them
    }

# -----------------------------
# CHAT UI — with conversation history
# -----------------------------

# ── Session state ──
if "all_conversations" not in st.session_state:
    st.session_state.all_conversations = []
if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_filters" not in st.session_state:
    st.session_state.last_filters = None
if "result_offset" not in st.session_state:
    st.session_state.result_offset = 0
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None
if "welcomed" not in st.session_state:
    st.session_state.welcomed = False

# ── Helpers ──
import uuid, datetime

def save_current_conv():
    """Persist current messages back into all_conversations."""
    if not st.session_state.messages:
        return
    cid = st.session_state.active_conv_id
    if cid is None:
        return
    for conv in st.session_state.all_conversations:
        if conv["id"] == cid:
            conv["messages"] = list(st.session_state.messages)
            return

def new_conversation():
    """Save current, then start fresh."""
    save_current_conv()
    cid = str(uuid.uuid4())[:8]
    tz_myt = datetime.timezone(datetime.timedelta(hours=8))
    now = datetime.datetime.now(tz=tz_myt).strftime("%b %d, %H:%M")
    st.session_state.all_conversations.append({
        "id":       cid,
        "title":    f"Chat {now}",
        "messages": []
    })
    st.session_state.active_conv_id   = cid
    st.session_state.messages         = []
    st.session_state.last_filters     = None
    st.session_state.result_offset    = 0
    st.session_state.selected_product = None
    st.session_state.welcomed         = False

def load_conversation(cid):
    """Switch to an existing conversation."""
    save_current_conv()
    for conv in st.session_state.all_conversations:
        if conv["id"] == cid:
            st.session_state.active_conv_id = cid
            st.session_state.messages       = list(conv["messages"])
            st.session_state.last_filters   = None
            st.session_state.result_offset  = 0
            st.session_state.selected_product = None
            return

def auto_title(messages):
    """Generate a title from the first user message."""
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], str):
            t = m["content"][:28]
            return t + ("…" if len(m["content"]) > 28 else "")
    return "New Chat"

# ── Bootstrap: ensure there's always one active conversation ──
if st.session_state.active_conv_id is None:
    new_conversation()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 💬 Conversations")
    if st.button("➕  New Chat", use_container_width=True, key="new_chat_btn"):
        new_conversation()
        st.rerun()

    st.markdown("---")
    convs = st.session_state.all_conversations
    for conv in reversed(convs):
        title = auto_title(conv["messages"]) if conv["messages"] else conv["title"]
        is_active = conv["id"] == st.session_state.active_conv_id
        label = f"{'▶ ' if is_active else ''}{title}"
        if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
            load_conversation(conv["id"])
            st.rerun()

    if len(convs) > 1:
        st.markdown("---")
        if st.button("🗑️ Delete this chat", use_container_width=True, key="del_conv_btn"):
            cid = st.session_state.active_conv_id
            st.session_state.all_conversations = [
                c for c in st.session_state.all_conversations if c["id"] != cid
            ]
            if st.session_state.all_conversations:
                load_conversation(st.session_state.all_conversations[-1]["id"])
            else:
                new_conversation()
            st.rerun()

# ── Welcome guide ──
if not st.session_state.welcomed:
    welcome_html = (
        "<div style='background:linear-gradient(135deg,#1a237e 0%,#283593 50%,#e53935 100%);"
        "border-radius:16px;padding:24px 28px;margin-bottom:20px;'>"
        "<h3 style='margin:0 0 6px 0;color:#ffffff;'>👋 Welcome to ShopAssist AI!</h3>"
        "<p style='margin:0 0 16px 0;color:#c5cae9;font-size:14px;'>Your smart Adidas product finder. Try asking:</p>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>"
        "<div style='background:rgba(255,255,255,0.1);border-radius:8px;padding:8px 12px;color:#ffffff;font-size:13px;'>👟 <b>Shoes</b><br><span style='color:#c5cae9;'>running shoes, slides, golf shoes</span></div>"
        "<div style='background:rgba(255,255,255,0.1);border-radius:8px;padding:8px 12px;color:#ffffff;font-size:13px;'>👕 <b>Clothing</b><br><span style='color:#c5cae9;'>hoodies, jacket, pants, tights</span></div>"
        "<div style='background:rgba(255,255,255,0.1);border-radius:8px;padding:8px 12px;color:#ffffff;font-size:13px;'>💰 <b>Price</b><br><span style='color:#c5cae9;'>shoes under 100 · between 30 and 60</span></div>"
        "<div style='background:rgba(255,255,255,0.1);border-radius:8px;padding:8px 12px;color:#ffffff;font-size:13px;'>🎨 <b>Color</b><br><span style='color:#c5cae9;'>black shoes · white hoodie · blue tee</span></div>"
        "<div style='background:rgba(255,255,255,0.1);border-radius:8px;padding:8px 12px;color:#ffffff;font-size:13px;'>👤 <b>Gender</b><br><span style='color:#c5cae9;'>women's running shoes · men's hoodie</span></div>"
        "<div style='background:rgba(255,255,255,0.1);border-radius:8px;padding:8px 12px;color:#ffffff;font-size:13px;'>⭐ <b>Sort</b><br><span style='color:#c5cae9;'>best shoes · cheap clothing</span></div>"
        "</div>"
        "<p style='margin:14px 0 0 0;color:#c5cae9;font-size:12px;'>"
        "💡 Click <b style='color:#fff;'>View →</b> for details &nbsp;·&nbsp; Say <b style='color:#fff;'>more</b> to load more results"
        "</p></div>"
    )
    st.markdown(welcome_html, unsafe_allow_html=True)
    st.markdown('<div class="welcome-btn">', unsafe_allow_html=True)
    if st.button("Got it! Let's shop 🛍️", key="welcome_dismiss"):
        st.session_state.welcomed = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Product detail view ──
if st.session_state.selected_product:
    display_product_detail(st.session_state.selected_product)
    st.stop()

# ── Render current conversation ──
chat_container = st.container()
with chat_container:
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if isinstance(content, dict):
                st.write(content["message"])
                data_value    = content["data"]
                response_type = content.get("type", "")
                saved_filters = content.get("filters", None)   # ← retrieve saved filters
                if response_type == "categories":
                    display_categories()
                elif response_type == "help":
                    show_examples()
                elif data_value is not None:
                    if isinstance(data_value, str) and data_value == "SHOW_EXAMPLES":
                        show_examples()
                    elif isinstance(data_value, pd.DataFrame):
                        display_products(
                            data_value,
                            label="Top Recommendations",
                            card_key_prefix=f"hist_{msg_idx}",
                            filters=saved_filters   # ← pass filters
                        )
            else:
                st.write(content)

# ── User input ──
user_input = st.chat_input("Ask for recommendations...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input)

    with st.chat_message("assistant"):
        if isinstance(response, dict):
            st.write(response["message"])
            data_value    = response["data"]
            response_type = response.get("type", "")
            response_filters = response.get("filters", None)   # ← get filters from response
            if response_type == "categories":
                display_categories()
            elif response_type == "help":
                show_examples()
            elif data_value is not None:
                if isinstance(data_value, str) and data_value == "SHOW_EXAMPLES":
                    show_examples()
                elif isinstance(data_value, pd.DataFrame):
                    display_products(
                        data_value,
                        label="Top Recommendations",
                        card_key_prefix=f"new_{len(st.session_state.messages)}",
                        filters=response_filters   # ← pass filters
                    )
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.write(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})

    # Auto-save after every turn
    save_current_conv()
    st.rerun()
