import pandas as pd
import streamlit as st

st.set_page_config(page_title="ShopAssist Chatbot", page_icon="🛍️")

st.title("🛍️ ShopAssist Chatbot")
st.markdown("Ask about products, prices, discounts, and recommendations!")

# Load dataset
df = pd.read_csv("diversified_ecommerce_dataset.csv")

# Clean column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Sidebar (nice UI)
st.sidebar.header("💡 Example Questions")
st.sidebar.write("""
- cheap products  
- best products  
- electronics  
- discount products  
- products in asia  
""")

# User input
user_input = st.text_input("💬 Ask your question:")

if user_input:
    user_input = user_input.lower()

    # Greeting
    if "hello" in user_input or "hi" in user_input:
        st.success("Hello! How can I help you today? 😊")

    # Cheap products
    elif "cheap" in user_input:
        cheap = df[df['price'] < df['price'].mean()]
        result = cheap.sort_values(by='price').head(5)

        st.subheader("💰 Budget Products")
        st.dataframe(result[['product_name', 'category', 'price']])

    # Best products (Popularity Index)
    elif "best" in user_input or "top" in user_input:
        best = df.sort_values(by='popularity_index', ascending=False).head(5)

        st.subheader("⭐ Top Popular Products")
        st.dataframe(best[['product_name', 'category', 'popularity_index']])

    # Discount products
    elif "discount" in user_input:
        discount = df.sort_values(by='discount', ascending=False).head(5)

        st.subheader("🔥 Best Discount Deals")
        st.dataframe(discount[['product_name', 'category', 'discount']])

    # Category search
    elif any(cat in user_input for cat in df['category'].str.lower().unique()):
        for cat in df['category'].str.lower().unique():
            if cat in user_input:
                results = df[df['category'].str.lower() == cat].head(5)

                st.subheader(f"📦 {cat.title()} Products")
                st.dataframe(results[['product_name', 'price', 'category']])
                break

    # Location-based
    elif "in" in user_input:
        location = user_input.split("in")[-1].strip()

        results = df[df['customer_location'].str.contains(location, case=False, na=False)]

        if not results.empty:
            st.subheader(f"🌍 Products popular in {location}")
            st.dataframe(results[['product_name', 'customer_location']].head(5))
        else:
            st.warning("No data found for that location.")

    # Stock alert
    elif "stock" in user_input:
        low_stock = df[df['stock_level'] < 20].head(5)

        st.subheader("⚠️ Low Stock Products")
        st.dataframe(low_stock[['product_name', 'stock_level']])

    else:
        st.info("Try asking about price, category, discount, or best products 😊")
