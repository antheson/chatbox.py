import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title="ShopAssist Chatbot", page_icon="🛍️")

st.title("🛍️ ShopAssist Chatbot")
st.markdown("Ask me to recommend products, find items, or explore brands!")

# Load dataset
df = pd.read_csv("diversified_ecommerce_dataset.csv")

# Clean column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# User input
user_input = st.text_input("💬 Your message:")

if user_input:
    user_input = user_input.lower()

    # Greeting
    if "hello" in user_input or "hi" in user_input:
        st.success("Hi! How can I help you today? 😊")

    # Cheap products
    elif "cheap" in user_input:
        cheap = df[df['price'] < df['price'].mean()]
        result = cheap.sample(1)

        st.subheader("💰 Budget Recommendation")
        st.dataframe(result[['product_name', 'brand', 'price']])

    # Best rated
    elif "best" in user_input:
        if 'rating' in df.columns:
            best = df.sort_values(by='rating', ascending=False).head(3)

            st.subheader("⭐ Top Rated Products")
            st.dataframe(best[['product_name', 'brand', 'rating']])
        else:
            st.warning("Rating data not available.")

    # Lipstick search (FIXED)
    elif "lipstick" in user_input:
        if 'sub_category' in df.columns:
            results = df[df['sub_category'].str.contains("lipstick", case=False, na=False)]

            st.subheader("💄 Lipstick Products")
            st.dataframe(results[['product_name', 'brand', 'price']].head(5))
        else:
            st.error("Sub-category column not found.")

    # Brand search
    elif "brand" in user_input:
        brand_name = user_input.split("brand")[-1].strip()

        results = df[df['brand'].str.contains(brand_name, case=False, na=False)]

        if not results.empty:
            st.subheader(f"🏷️ Products from {brand_name}")
            st.dataframe(results[['product_name', 'price']].head(5))
        else:
            st.warning("No products found for that brand.")

    # Default
    else:
        st.info("Try asking things like:\n- 'cheap products'\n- 'best items'\n- 'lipstick'\n- 'brand loreal'")
