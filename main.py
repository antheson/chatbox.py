import pandas as pd
import streamlit as st
import os

st.title("🛍️ ShopAssist Chatbot")

# Debug: check files
st.write("Files in directory:", os.listdir())

# Load dataset
df = pd.read_csv("diversified_ecommerce_dataset.csv")

# Clean columns
df.columns = df.columns.str.lower().str.replace(" ", "_")

user_input = st.text_input("Ask me about products:")

if user_input:
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        st.write("Bot: Hi! How can I help you today?")

    elif "cheap" in user_input:
        cheap = df[df['price'] < df['price'].mean()]
        result = cheap.sample(1)
        st.write("Bot: Budget product:")
        st.dataframe(result[['product_name', 'brand', 'price']])

    elif "best" in user_input:
        if 'rating' in df.columns:
            best = df.sort_values(by='rating', ascending=False).head(1)
            st.write("Bot: Best product:")
            st.dataframe(best[['product_name', 'brand', 'rating']])
        else:
            st.write("Bot: No rating data available")

    elif "lipstick" in user_input:
        results = df[df['subcategory'].str.contains("lipstick", case=False, na=False)]
        st.write("Bot: Lipstick products:")
        st.dataframe(results[['product_name', 'brand', 'price']].head(3))

    elif "brand" in user_input:
        brand_name = user_input.split("brand")[-1].strip()
        results = df[df['brand'].str.contains(brand_name, case=False, na=False)]

        if not results.empty:
            st.write(f"Bot: Products from {brand_name}:")
            st.dataframe(results[['product_name', 'price']].head(3))
        else:
            st.write("Bot: No products found")

    else:
        st.write("Bot: Sorry, I didn't understand.")
