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

st.markdown("""
<style>
    .stApp { background-color: #f7f8fc; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
    section[data-testid="stSidebar"] h2 { color: #ffffff !important; letter-spacing: 0.5px; }
    section[data-testid="stSidebar"] hr { border-color: #2e3a5c !important; }
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
    df.columns = df.columns.str.lower().str.replace(" ", "_")
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
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'original_price' in df.columns:
        df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
    if 'price' in df.columns and 'original_price' in df.columns:
        df['price'] = df['price'].fillna(df['original_price'])
    df['price'] = df['price'].fillna(0)
    df['product_name'] = df['product_name'].fillna('Unknown Product')
    df['category'] = df['category'].fillna('Uncategorized')
    df['color'] = df['color'].fillna('Multiple Colors')
    df = df[df['category'] != 'Uncategorized']

    # ── FIX 1: Expanded gender extraction to cover non-Men/Women breadcrumbs ──
    # Breadcrumbs like "Soccer/Shoes", "Running/Shoes", "Originals/Clothing",
    # "Essentials/Shoes", "Five Ten/Shoes", "Training/Accessories", "Swim/Shoes"
    # are genuinely gender-neutral — keep them as 'Unisex'.
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
            return 'Unisex'   # Soccer/, Running/, Originals/, Essentials/, etc.

    df['gender'] = df['breadcrumbs'].apply(extract_gender)

    df['image_url'] = df['images'].apply(
        lambda x: str(x).split('~')[0].strip() if pd.notna(x) else ''
    )
    return df

df = load_data()

ALL_CATEGORIES = df['category'].dropna().unique().tolist()
ALL_COLORS = df['color'].dropna().unique().tolist()

# -----------------------------
# TYPO CORRECTION
# -----------------------------
def correct_typo(word, word_list, cutoff=0.8):
    if word in word_list:
        return word
    matches = get_close_matches(word, word_list, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    return word

def correct_category_typo(user_input, categories):
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
    # ── Phase 1: phrase-level corrections (multi-word typos) ──
    phrase_corrections = {
        'show more': ['show mroe', 'show moer', 'show mor', 'show mre', 'shwo more',
                      'sohw more', 'show morre', 'shoow more', 'show moore',
                      'sohw mroe', 'shwo mroe'],
        'see more':  ['see mroe', 'see moer', 'see mor', 'se more', 'see mre'],
        'show me more': ['show me mroe', 'show me moer'],
        'load more': ['laod more', 'loda more', 'load mroe'],
    }
    text = user_input.lower().strip()
    # First pass: exact full-message match (highest priority, no substring risk)
    for canonical, typos in phrase_corrections.items():
        if text in typos:
            return canonical
    # Second pass: substring replacement — sort longer typos first to avoid
    # partial matches (e.g. 'show mor' must not eat 'show morre' before it matches).
    # Skip if canonical is already present in text to avoid double-replacement.
    for canonical, typos in phrase_corrections.items():
        if canonical in text:
            continue
        for typo in sorted(typos, key=len, reverse=True):
            if typo in text:
                text = text.replace(typo, canonical)
                break  # only one replacement per canonical per pass

    # ── Phase 2: word-level corrections ──
    word_corrections = {
        'hello':       ['hllo', 'helo', 'hellp', 'hallow', 'halo'],
        'help':        ['halp', 'hlp', 'hepl', 'helpp'],
        'cheap':       ['cheep', 'chap', 'chep', 'cheapp'],
        'best':        ['bests', 'besst', 'bist'],
        'expensive':   ['expensiv', 'expen', 'costly'],
        'categories':  ['catagories', 'categries', 'catgories', 'categorys'],
        'recommend':   ['recomend', 'reccomend', 'rekomend', 'recommanded'],
        'thank':       ['thx', 'thankyou', 'tq', 'ty'],
        'shoes':       ['shoees', 'sheos', 'shoess', 'shos', 'shose', 'suoes', 'suoe'],
        'clothing':    ['vlothing', 'cloting', 'clohting', 'clothng', 'clthing',
                        'clothin', 'clotthing', 'clething', 'cloding', 'clohing', 'cloathing'],
        'accessories': ['acessories', 'accesories', 'accessorys', 'acessory',
                        'accessoires', 'accesory'],
        'slides':      ['slids', 'sldies', 'sldes'],
        'sandals':     ['sandel', 'sandels', 'sandlas'],
        'running':     ['runing', 'runnin', 'runninng'],
        'casual':      ['cazual', 'casaul', 'casuel', 'causal'],
        'unisex':      ['unisec', 'unisecs', 'unisexs', 'unisez', 'unisek',
                        'uniisex', 'unisexe', 'unixex', 'unisax', 'unisix', 'unesex'],
        'more':        ['mroe', 'moer', 'mre'],
        'next':        ['nxt', 'nexy', 'nect'],
        'show':        ['shwo', 'sohw', 'shoow'],
    }
    words = text.split()
    corrected_words = []
    for word in words:
        corrected = word
        for keyword, variations in word_corrections.items():
            if word in variations:
                corrected = keyword
                break
        corrected_words.append(corrected)
    return ' '.join(corrected_words)

# -----------------------------
# TRAINING DATA
# -----------------------------
training_data = [
    ("hello", "greeting"), ("hi", "greeting"), ("hey", "greeting"),
    ("what can you do", "help"), ("help me", "help"), ("how to use", "help"),
    ("what can i ask", "help"),
    ("recommend product", "recommend"), ("suggest something", "recommend"),
    ("cheap products", "cheap"), ("low price items", "cheap"),
    ("best products", "best"), ("top products", "best"), ("most popular", "best"),
    ("expensive products", "expensive"), ("most expensive", "expensive"), ("premium products", "expensive"),
    ("show categories", "categories"), ("what categories", "categories"), ("list categories", "categories"),
    ("thank", "thanks"), ("thanks", "thanks"), ("thank you", "thanks")
]

X = [x[0] for x in training_data]
y = [x[1] for x in training_data]
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X_vector, y)

def predict_intent(text):
    corrected_text = correct_intent_typo(text)
    return model.predict(vectorizer.transform([corrected_text.lower()]))[0]

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
# SMART EXPLANATION GENERATOR
# -----------------------------
def generate_explanation(row, filters):
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

    if max_price and price and price <= max_price:
        pills.append("✔ In budget")
    elif min_price and max_price and price:
        pills.append("✔ In price range")
    if filters.get('color') and color and filters['color'].lower() in color.lower():
        pills.append(f"✔ {color} color")
    if filters.get('gender') and gender == filters['gender']:
        pills.append(f"✔ For {gender}")
    if filters.get('subcategory'):
        pills.append(f"✔ {filters['subcategory'].capitalize()}")
    try:
        r = float(rating)
        if r >= 4.5:
            pills.append(f"⭐ {r}/5 rated")
    except Exception:
        pass
    try:
        orig = float(orig_price) if orig_price else 0
        cur  = float(price) if price else 0
        if orig > 0 and cur > 0 and orig > cur:
            pct = round((orig - cur) / orig * 100)
            if pct >= 10:
                pills.append(f"🔥 {pct}% off")
    except Exception:
        pass
    if intent == 'cheap' and price:
        pills.append(f"💰 ${price:.0f}")
    elif intent == 'expensive' and price:
        pills.append("💎 Premium")
    if availability == 'InStock' and len(pills) == 0:
        pills.append("✔ In stock")
    if not pills:
        try:
            r = float(rating)
            if r > 0:
                pills.append(f"⭐ {r}/5")
        except Exception:
            pass
        pills.append("✔ Matches search")
    return pills[:3]

def render_explanation(pills):
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
    rating_str  = (f"{'⭐' * int(round(float(rating)))}{'☆' * (5 - int(round(float(rating))))} ({rating}/5)") if rating and float(rating) > 0 else "No rating"
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
# DISPLAY PRODUCTS
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
# DISPLAY CATEGORIES
# -----------------------------
def display_categories():
    categories = sorted(df['category'].dropna().unique())
    st.subheader("📚 Available Product Categories")
    cols = st.columns(3)
    for idx, category in enumerate(categories):
        with cols[idx % 3]:
            product_count = len(df[df['category'] == category])
            unique_colors = df[df['category'] == category]['color'].nunique()
            st.write(f"• **{category}** ({product_count} products, {unique_colors} colors)")
    st.info(f"💡 Total: {len(categories)} categories available")
    with st.expander("🔍 Want to see sample products from a category?"):
        selected_category = st.selectbox("Choose a category:", categories)
        if selected_category:
            sample_products = df[df['category'] == selected_category].head(5)
            st.write(f"**Sample products in {selected_category}:**")
            for _, product in sample_products.iterrows():
                color_info = f" ({product['color']})" if product['color'] != 'Multiple Colors' else ""
                st.write(f"• {product['product_name']}{color_info} - ${product.get('price', 0):.2f}")

# -----------------------------
# EXTRACT FILTERS  ← ALL FIXES ARE HERE
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

    # ── FIX 2: Strict price parsing — only parse numbers that are paired with
    #    a clear price keyword or explicit connector ("between/to/under/over").
    #    Bare isolated numbers like "100 200 shoes" are now rejected.
    # ──────────────────────────────────────────────────────────────────────────

    # Pattern 1: "between X and Y" or "X-Y"
    between_match = re.search(r'between\s+(\d+)\s*(?:and|-)\s*(\d+)', text_lower)
    if between_match:
        filters['min_price'] = float(between_match.group(1))
        filters['max_price'] = float(between_match.group(2))
    else:
        # Pattern 2: "X to Y" (requires the word "to")
        to_match = re.search(r'(\d+)\s+to\s+(\d+)', text_lower)
        if to_match:
            filters['min_price'] = float(to_match.group(1))
            filters['max_price'] = float(to_match.group(2))
        else:
            # Pattern 3: explicit directional keywords only
            under_match = re.search(r'(?:under|below|less\s+than)\s+(\d+)', text_lower)
            if under_match:
                filters['max_price'] = float(under_match.group(1))

            above_match = re.search(r'(?:above|over|more\s+than)\s+(\d+)', text_lower)
            if above_match:
                filters['min_price'] = float(above_match.group(1))

            # Pattern 4: "$X" explicit dollar sign (e.g. "shoes $80")
            if filters['min_price'] is None and filters['max_price'] is None:
                dollar_match = re.search(r'\$(\d+)', text_lower)
                if dollar_match:
                    filters['max_price'] = float(dollar_match.group(1))

            # ── NO bare-number fallback — "100 200 shoes" should not parse as price ──

    # ── CATEGORY DETECTION ──────────────────────────────────────────────────
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

    # ── FIX 3: Color detection — treat unknown colors as explicit "not found"
    #    so the chatbot refuses to show results instead of ignoring the color.
    # ──────────────────────────────────────────────────────────────────────────
    # Colors that exist in the dataset
    VALID_COLOR_MAP = {
        'black': 'Black', 'white': 'White', 'blue': 'Blue', 'red': 'Red',
        'green': 'Green', 'yellow': 'Yellow', 'pink': 'Pink', 'purple': 'Purple',
        'grey': 'Grey', 'gray': 'Grey', 'beige': 'Beige', 'gold': 'Gold',
        'burgundy': 'Burgundy', 'multicolor': 'Multicolor',
        'multi color': 'Multicolor', 'colorful': 'Multicolor',
        'multi': 'Multicolor',
        'orange': 'Red',   # closest available
        'brown': 'Beige',  # closest available
    }
    # Colors explicitly NOT in the dataset — always show "not found"
    INVALID_COLORS = {
        'rainbow', 'transparent', 'invisible', 'clear', 'holographic',
        'neon', 'glow', 'glitter', 'sparkle', 'chrome', 'silver',
        'copper', 'navy', 'teal', 'cyan', 'magenta', 'lavender',
        'ivory', 'cream', 'tan', 'olive', 'maroon',
    }

    color_found = False
    # Check invalid colors first (they take priority)
    for inv_color in INVALID_COLORS:
        if re.search(rf'\b{re.escape(inv_color)}\b', text_lower):
            filters['color'] = None           # no matching color
            filters['color_searched'] = inv_color
            filters['color_not_found'] = True  # explicit not-found flag
            color_found = True
            break

    if not color_found:
        for user_color, dataset_color in VALID_COLOR_MAP.items():
            if re.search(rf'\b{re.escape(user_color)}\b', text_lower):
                filters['color'] = dataset_color
                filters['color_searched'] = user_color
                color_found = True
                break

    # ── SUBCATEGORY DETECTION ──────────────────────────────────────────────
    subcategory_map = {
        'running':    {'user_terms': ['running', 'run', 'jogging', 'jog'],
                       'name_keywords': ['run', 'boost', 'pureboost', 'ultraboost', 'supernova', 'eq21', 'tensor', 'fluidflash']},
        'slides':     {'user_terms': ['slides', 'slide', 'sandal', 'sandals', 'flip flop', 'flip-flop', 'slippers'],
                       'name_keywords': ['slide', 'sandal', 'flip', 'mule', 'adilette']},
        'soccer':     {'user_terms': ['soccer', 'football', 'cleat', 'cleats', 'turf'],
                       'name_keywords': ['copa', 'predator', 'ghosted', 'cleat', 'turf', 'firm ground']},
        'golf':       {'user_terms': ['golf'], 'name_keywords': ['golf']},
        'climbing':   {'user_terms': ['climbing', 'climb', 'boulder'],
                       'name_keywords': ['climbing', 'hiangle', 'five ten']},
        'cycling':    {'user_terms': ['cycling', 'bike', 'mountain bike', 'mtb'],
                       'name_keywords': ['bike', 'kestrel', 'five ten']},
        'basketball': {'user_terms': ['basketball', 'court'], 'name_keywords': ['hoops']},
        'casual':     {'user_terms': ['casual', 'lifestyle', 'everyday', 'classic', 'street'],
                       'name_keywords': ['superstar', 'samba', 'nizza', 'hamburg', 'ozweego', 'ozelia',
                                         'postmove', 'grand court', 'advantage', 'retrorun', 'sambarose', 'bryony']},
        'training':   {'user_terms': ['training', 'gym', 'workout', 'cross training', 'crossfit'],
                       'name_keywords': ['racer', 'swift', 'climacool', 'multix', 'puremotion', 'futurenatural']},
        'hiking':     {'user_terms': ['hiking', 'hike', 'trail', 'outdoor', 'trekking'],
                       'name_keywords': ['hike', 'trail', 'outdoor', 'kestrel']},
        'hoodie':     {'user_terms': ['hoodie', 'hoodies', 'hoddie', 'hoody', 'zip up', 'zip-up'],
                       'name_keywords': ['hoodie']},
        'tee':        {'user_terms': ['tee', 'tees', 't-shirt', 'tshirt', 't shirt', 'shirt', 'polo'],
                       'name_keywords': ['tee', 'polo']},
        'sweatshirt': {'user_terms': ['sweatshirt', 'sweatshirts', 'sweater', 'crewneck', 'crew neck'],
                       'name_keywords': ['sweatshirt']},
        'jacket':     {'user_terms': ['jacket', 'jackets', 'windbreaker', 'wind breaker'],
                       'name_keywords': ['jacket', 'windbreaker']},
        'pants':      {'user_terms': ['pants', 'trousers', 'joggers', 'track pants', 'trackpants'],
                       'name_keywords': ['pants']},
        'shorts':     {'user_terms': ['shorts', 'short'], 'name_keywords': ['shorts']},
        'tights':     {'user_terms': ['tights', 'leggings', 'leggins', 'tight'], 'name_keywords': ['tights']},
        'dress':      {'user_terms': ['dress', 'dresses'], 'name_keywords': ['dress']},
        'tank top':   {'user_terms': ['tank top', 'tank', 'crop top', 'tanktop'],
                       'name_keywords': ['tank top', 'crop top']},
        'jersey':     {'user_terms': ['jersey', 'jerseys', 'kit'], 'name_keywords': ['jersey']},
        'tracksuit':  {'user_terms': ['tracksuit', 'track suit', 'tracksuits', 'sst'],
                       'name_keywords': ['track suit', 'sst set']},
        'swimwear':   {'user_terms': ['swimwear', 'swimsuit', 'swim shorts', 'swimming'],
                       'name_keywords': ['swimsuit', 'swim shorts']},
    }
    clothing_subcats = {
        'hoodie', 'tee', 'sweatshirt', 'jacket', 'pants', 'shorts',
        'tights', 'dress', 'tank top', 'jersey', 'tracksuit', 'swimwear'
    }

    filters['subcategory'] = None
    filters['subcategory_name_keywords'] = None
    filters['subcategory_category'] = None
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
        filters['subcategory'] = matched_subcats[0][0]
        filters['subcategory_name_keywords'] = matched_subcats[0][1]
        filters['subcategory_category'] = matched_subcats[0][2]
        filters['all_matched_subcats'] = matched_subcats
    else:
        filters['all_matched_subcats'] = None

    # ── FIX 4: Gender detection — added "unisex" keyword explicitly ──────────
    filters['gender'] = None
    if re.search(r'\bunisex\b', text_lower):
        filters['gender'] = 'Unisex'
    elif re.search(r"\bwomen'?s?\b|\bfemale\b|\bladies\b|\bgirl\b|\bgirls\b", text_lower):
        filters['gender'] = 'Women'
    elif re.search(r"\bmen'?s?\b|\bmale\b|\bguy\b|\bguys\b|\bboy\b", text_lower):
        filters['gender'] = 'Men'
    elif re.search(r'\bkids?\b|\bchildren\b|\bjunior\b', text_lower):
        filters['gender'] = 'Kids'

    # ── INTENT DETECTION ────────────────────────────────────────────────────
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
    elif filters['min_price'] or filters['max_price']:
        filters['intent'] = 'price_range'

    return filters


# -----------------------------
# PRODUCT NAME SEARCH
# -----------------------------
def search_by_product_name(user_input):
    trigger_phrases = [
        'find ', 'search for ', 'search ', 'show me ', 'look for ',
        'looking for ', 'do you have ', 'do you sell ', 'i want ',
        'i need ', 'get me ', 'show ', 'any '
    ]
    query = user_input.lower().strip()
    for phrase in trigger_phrases:
        if query.startswith(phrase):
            query = query[len(phrase):].strip()

    if len(query) < 4:
        return None

    filter_words = {
        'shoes', 'shoe', 'clothing', 'clothes', 'accessories', 'accessory',
        'cheap', 'expensive', 'best', 'top', 'budget', 'premium', 'running',
        'casual', 'slides', 'sandals', 'hoodie', 'jacket', 'pants', 'shorts',
        'black', 'white', 'blue', 'red', 'green', 'pink', 'grey', 'gray',
        'men', 'women', 'kids', 'under', 'above', 'between', 'more',
        'unisex', 'unisec', 'unisecs', 'unisexs', 'unisez', 'unisek',
        'uniisex', 'unisexe', 'unixex', 'unisax', 'unisix', 'unesex',
        'rainbow', 'transparent', 'invisible',
        # pagination phrases — never treat as product names
        'show', 'next', 'see', 'give', 'load', 'results', 'page', 'please',
        'show mroe', 'show moer', 'show mor', 'shwo more', 'sohw more',
        'see mroe', 'see moer', 'nxt', 'nextt', 'nexy', 'nect',
    }
    query_words = set(query.split())
    if query_words.issubset(filter_words):
        return None

    all_names_lower = df['product_name'].dropna().str.lower().tolist()

    exact_mask = df['product_name'].str.lower().str.contains(re.escape(query), na=False)
    if exact_mask.any():
        return df[exact_mask]

    words = [w for w in query.split() if len(w) > 2 and w not in filter_words]
    if len(words) >= 2:
        mask = pd.Series([True] * len(df), index=df.index)
        for w in words:
            mask = mask & df['product_name'].str.lower().str.contains(re.escape(w), na=False)
        if mask.any():
            return df[mask]

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
    categories_list = [cat.lower() for cat in df['category'].dropna().unique()]
    corrected_input = correct_category_typo(user_input, categories_list)
    corrected_input = correct_intent_typo(corrected_input)

    # ── GUARD: reject invalid colors BEFORE name search runs ──────────────────
    # Without this, "invisible shoes" fuzzy-matches product names and leaks results.
    INVALID_COLORS = {
        'rainbow', 'transparent', 'invisible', 'clear', 'holographic',
        'neon', 'glow', 'glitter', 'sparkle', 'chrome', 'silver',
        'copper', 'navy', 'teal', 'cyan', 'magenta', 'lavender',
        'ivory', 'cream', 'tan', 'olive', 'maroon',
    }
    text_lower_raw = user_input.lower()
    for inv_color in INVALID_COLORS:
        if re.search(rf'\b{re.escape(inv_color)}\b', text_lower_raw):
            available = 'black, white, blue, red, green, yellow, pink, purple, grey, beige, gold, burgundy, multicolor'
            dummy_filters = {'category': None, 'color': None, 'min_price': None,
                             'max_price': None, 'intent': 'recommend',
                             'subcategory': None, 'gender': None}
            return {
                "type": "text",
                "message": f"Sorry, we don't carry any '{inv_color}' products. 😊\nAvailable colors: {available}.",
                "data": None,
                "filters": dummy_filters
            }
    # ─────────────────────────────────────────────────────────────────────────

    # ── SHORT-CIRCUIT: if corrected input is a pagination phrase, skip name search ──
    _pre_check = corrected_input.lower().strip()
    _PURE_MORE_EARLY = {
        'more', 'next', 'show more', 'see more', 'give more',
        'load more', 'more results', 'next page', 'more please',
        'show me more', 'next results',
    }
    if _pre_check in _PURE_MORE_EARLY or re.fullmatch(r'more[.!?]*', _pre_check):
        name_results = None
    else:
        name_results = search_by_product_name(corrected_input)
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
        st.session_state.last_had_results = True
        return {
            "type": "dataframe",
            "message": msg,
            "data": name_results[cols_needed].reset_index(drop=True),
            "filters": filters
        }

    # ── PAGINATION DETECTION ──────────────────────────────────────────────────
    # Only treat as "show more" if the message is PURELY a pagination request
    # with no new filter words. Prevents "more expensive shoes" or
    # "show more women shoes" from paginating instead of running a fresh search.
    _input_stripped = corrected_input.lower().strip()

    PURE_MORE_PHRASES = {
        'more', 'next', 'show more', 'see more', 'give more',
        'load more', 'more results', 'next page', 'more please',
        'show me more', 'next results',
    }
    is_pure_more = _input_stripped in PURE_MORE_PHRASES
    if not is_pure_more and re.fullmatch(r'more[.!?]*', _input_stripped):
        is_pure_more = True

    if is_pure_more:
        if st.session_state.last_had_results:
            st.session_state.result_offset += 5
            filters = st.session_state.last_filters
        else:
            return {
                "type": "text",
                "message": "There's nothing to show more of yet. Search for some products first — try 'shoes under 100' or 'best running shoes'! 😊",
                "data": None,
                "filters": {'category': None, 'color': None, 'min_price': None,
                             'max_price': None, 'intent': 'recommend',
                             'subcategory': None, 'gender': None}
            }
    else:
        filters = extract_filters(user_input)
        st.session_state.last_filters = filters
        st.session_state.last_had_results = False
        st.session_state.result_offset = 0
    # ─────────────────────────────────────────────────────────────────────────

    model_intent = "unknown"
    if not filters['category'] and not filters['color'] and not filters['min_price'] and not filters['max_price'] and not filters.get('subcategory'):
        try:
            model_intent = predict_intent(corrected_input)
        except:
            model_intent = "unknown"

    text_lower = corrected_input.lower()

    if any(phrase in text_lower for phrase in ["show categories", "what categories", "list categories", "all categories", "available categories"]):
        return {"type": "categories", "message": "Here are all the product categories available! 🛍️", "data": None, "filters": filters}

    if model_intent == "greeting":
        return {"type": "text", "message": "Hi there! 👋 I'm your shopping assistant.\n\nYou can ask me to recommend products based on price, category, color, or rating!", "data": None, "filters": filters}

    if not filters['category'] and not filters['color'] and not filters['min_price'] and not filters['max_price'] and not filters.get('subcategory') and not filters.get('gender'):
        if model_intent == "thanks":
            import random
            return {"type": "text", "message": random.choice(["You're very welcome! 😊", "My pleasure! 🛍️", "Anytime! 🙌"]), "data": None, "filters": filters}
        if model_intent == "help":
            return {"type": "help", "message": "Here are some things you can ask me 😊", "data": None, "filters": filters}

    limit  = 5
    offset = st.session_state.result_offset
    result = df.copy()

    # ── FIX 3 (continued): Reject queries with non-existent colors immediately ──
    if filters.get('color_not_found'):
        searched = filters.get('color_searched', 'that color')
        available = 'black, white, blue, red, green, yellow, pink, purple, grey, beige, gold, burgundy, multicolor'
        return {
            "type": "text",
            "message": f"Sorry, we don't carry any '{searched}' products. 😊\nAvailable colors: {available}.",
            "data": None,
            "filters": filters
        }

    if filters['category']:
        result = result[result['category'].str.contains(filters['category'], case=False, na=False)]

    if filters.get('color_searched') and filters['color'] is None and not filters.get('color_not_found'):
        searched = filters['color_searched']
        available = 'black, white, blue, red, green, yellow, pink, purple, grey, beige, gold, burgundy, multicolor'
        return {"type": "text", "message": f"Sorry, we don't have any '{searched}' products. 😊 Available colors: {available}.", "data": None, "filters": filters}

    if filters['color']:
        result = result[result['color'].str.contains(filters['color'], case=False, na=False)]

    # ── FIX 4 (continued): Apply unisex gender filter strictly ──────────────
    if filters.get('gender'):
        result = result[result['gender'] == filters['gender']]

    if filters['min_price']:
        result = result[result['price'] >= filters['min_price']]
    if filters['max_price']:
        result = result[result['price'] <= filters['max_price']]

    def apply_single_subcat(base_df, name_keywords, subcat_category, main_category_filter):
        pattern = '|'.join(re.escape(kw) for kw in name_keywords)
        sub_result = base_df[base_df['product_name'].str.contains(pattern, case=False, na=False)]
        if sub_result.empty:
            sub_result = base_df
        if subcat_category and not main_category_filter:
            sub_result = sub_result[sub_result['category'].str.contains(subcat_category, case=False, na=False)]
        return sub_result

    all_subcats = filters.get('all_matched_subcats')
    if all_subcats and len(all_subcats) > 1:
        frames = []
        subcat_names = []
        for (subcat_name, name_kws, subcat_cat) in all_subcats:
            part = apply_single_subcat(result, name_kws, subcat_cat, filters['category'])
            if not part.empty:
                frames.append(part)
                subcat_names.append(subcat_name)
        if frames:
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

    if result.empty:
        conditions = []
        if filters.get('gender'):
            conditions.append(f"gender '{filters['gender']}'")
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

        return {"type": "text", "message": msg, "data": None, "filters": filters}

    is_multi_combo = bool(filters.get('all_matched_subcats') and len(filters['all_matched_subcats']) > 1)

    if filters['intent'] == 'expensive':
        sorted_result = result[result['price'] > 0].sort_values('price', ascending=False)
        msg = "Here are the most expensive products"
    elif filters['intent'] == 'cheap':
        sorted_result = result[result['price'] > 0].sort_values('price')
        msg = "Here are budget-friendly products"
    elif filters['intent'] == 'best' and not is_multi_combo:
        sorted_result = result[result['popularity_index'] > 0].sort_values('popularity_index', ascending=False)
        msg = "Here are the highest-rated products"
    elif filters['intent'] == 'price_range':
        sorted_result = result[result['price'] > 0].sort_values('price')
        if filters['min_price'] and filters['max_price']:
            msg = f"Here are products between ${filters['min_price']:.0f} and ${filters['max_price']:.0f}"
        elif filters['max_price']:
            msg = f"Here are products under ${filters['max_price']:.0f}"
        elif filters['min_price']:
            msg = f"Here are products above ${filters['min_price']:.0f}"
        else:
            msg = "Here are recommended products"
    else:
        if is_multi_combo:
            sorted_result = result
            msg = "Here's a random mix of products"
        else:
            sorted_result = result[result['popularity_index'] > 0].sort_values('popularity_index', ascending=False)
            msg = "Here are recommended products"

    if filters.get('multi_subcat_limit_applied') and offset == 0:
        page_result = result
    else:
        page_result = sorted_result.iloc[offset:offset + limit]

    if page_result.empty:
        st.session_state.result_offset = max(0, offset - limit)
        return {"type": "text", "message": "No more results to show! Try a different search 😊", "data": None, "filters": filters}

    total_shown     = offset + len(page_result)
    total_available = len(sorted_result)
    msg += f" (showing {offset + 1}–{total_shown} of {total_available})"

    if filters.get('gender'):
        msg += f" · {filters['gender']}"
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

    st.session_state.last_had_results = True
    return {
        "type": "dataframe",
        "message": msg,
        "data": page_result[['product_name', 'category', 'price', 'color',
                               'popularity_index', 'review_count', 'gender',
                               'availability', 'description', 'image_url', 'original_price']],
        "filters": filters
    }


# -----------------------------
# CHAT UI
# -----------------------------
if "all_conversations" not in st.session_state:
    st.session_state.all_conversations = []
if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_filters" not in st.session_state:
    st.session_state.last_filters = None
if "last_had_results" not in st.session_state:
    st.session_state.last_had_results = False
if "result_offset" not in st.session_state:
    st.session_state.result_offset = 0
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None
if "welcomed" not in st.session_state:
    st.session_state.welcomed = False

import uuid, datetime

def save_current_conv():
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
    save_current_conv()
    cid = str(uuid.uuid4())[:8]
    tz_myt = datetime.timezone(datetime.timedelta(hours=8))
    now = datetime.datetime.now(tz=tz_myt).strftime("%b %d, %H:%M")
    st.session_state.all_conversations.append({"id": cid, "title": f"Chat {now}", "messages": []})
    st.session_state.active_conv_id   = cid
    st.session_state.messages         = []
    st.session_state.last_filters     = None
    st.session_state.last_had_results  = False
    st.session_state.result_offset    = 0
    st.session_state.selected_product = None
    st.session_state.welcomed         = False

def load_conversation(cid):
    save_current_conv()
    for conv in st.session_state.all_conversations:
        if conv["id"] == cid:
            st.session_state.active_conv_id   = cid
            st.session_state.messages         = list(conv["messages"])
            st.session_state.last_filters     = None
            st.session_state.last_had_results  = False
            st.session_state.result_offset    = 0
            st.session_state.selected_product = None
            return

def auto_title(messages):
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], str):
            t = m["content"][:28]
            return t + ("…" if len(m["content"]) > 28 else "")
    return "New Chat"

if st.session_state.active_conv_id is None:
    new_conversation()

with st.sidebar:
    st.markdown("## 💬 Conversations")
    if st.button("➕  New Chat", use_container_width=True, key="new_chat_btn"):
        new_conversation()
        st.rerun()
    st.markdown("---")
    for conv in reversed(st.session_state.all_conversations):
        title = auto_title(conv["messages"]) if conv["messages"] else conv["title"]
        is_active = conv["id"] == st.session_state.active_conv_id
        label = f"{'▶ ' if is_active else ''}{title}"
        if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
            load_conversation(conv["id"])
            st.rerun()
    if len(st.session_state.all_conversations) > 1:
        st.markdown("---")
        if st.button("🗑️ Delete this chat", use_container_width=True, key="del_conv_btn"):
            cid = st.session_state.active_conv_id
            st.session_state.all_conversations = [c for c in st.session_state.all_conversations if c["id"] != cid]
            if st.session_state.all_conversations:
                load_conversation(st.session_state.all_conversations[-1]["id"])
            else:
                new_conversation()
            st.rerun()

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
        "<div style='background:rgba(255,255,255,0.1);border-radius:8px;padding:8px 12px;color:#ffffff;font-size:13px;'>👤 <b>Gender</b><br><span style='color:#c5cae9;'>women's running shoes · men's hoodie · unisex</span></div>"
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

if st.session_state.selected_product:
    display_product_detail(st.session_state.selected_product)
    st.stop()

chat_container = st.container()
with chat_container:
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if isinstance(content, dict):
                st.write(content["message"])
                data_value    = content["data"]
                response_type = content.get("type", "")
                saved_filters = content.get("filters", None)
                if response_type == "categories":
                    display_categories()
                elif response_type == "help":
                    show_examples()
                elif data_value is not None:
                    if isinstance(data_value, str) and data_value == "SHOW_EXAMPLES":
                        show_examples()
                    elif isinstance(data_value, pd.DataFrame):
                        display_products(data_value, label="Top Recommendations",
                                         card_key_prefix=f"hist_{msg_idx}", filters=saved_filters)
            else:
                st.write(content)

user_input = st.chat_input("Ask for recommendations...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input)

    with st.chat_message("assistant"):
        if isinstance(response, dict):
            st.write(response["message"])
            data_value       = response["data"]
            response_type    = response.get("type", "")
            response_filters = response.get("filters", None)
            if response_type == "categories":
                display_categories()
            elif response_type == "help":
                show_examples()
            elif data_value is not None:
                if isinstance(data_value, str) and data_value == "SHOW_EXAMPLES":
                    show_examples()
                elif isinstance(data_value, pd.DataFrame):
                    display_products(data_value, label="Top Recommendations",
                                     card_key_prefix=f"new_{len(st.session_state.messages)}",
                                     filters=response_filters)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.write(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})

    save_current_conv()
    st.rerun()
