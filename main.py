import pandas as pd
import random

# Load dataset
df = pd.read_csv("E-commerce comestic dataset.csv")

# Clean column names (important!)
df.columns = df.columns.str.lower()

# -----------------------------
# Simple Chatbot Logic
# -----------------------------
def chatbot():
    print("🤖 ShopAssist Bot: Hello! Ask me about products 😊")

    while True:
        user_input = input("You: ").lower()

        # Exit
        if user_input in ["bye", "exit"]:
            print("Bot: Goodbye!")
            break

        # Greeting
        elif "hello" in user_input or "hi" in user_input:
            print("Bot: Hi! How can I help you today?")

        # Recommend cheap product
        elif "cheap" in user_input:
            cheap_products = df[df['price'] < df['price'].mean()]
            result = cheap_products.sample(1)

            print("Bot: I recommend this budget product:")
            print(result[['product name', 'brand', 'price']])

        # Best rated product
        elif "best" in user_input or "top" in user_input:
            best = df.sort_values(by='rating', ascending=False).head(1)

            print("Bot: Here is the best rated product:")
            print(best[['product name', 'brand', 'rating']])

        # Search by category
        elif "lipstick" in user_input:
            results = df[df['subcategory'].str.contains("lipstick", case=False, na=False)]

            print("Bot: Here are some lipsticks:")
            print(results[['product name', 'brand', 'price']].head(3))

        # Search by brand
        elif "brand" in user_input:
            brand_name = user_input.split("brand")[-1].strip()
            results = df[df['brand'].str.contains(brand_name, case=False, na=False)]

            print(f"Bot: Products from {brand_name}:")
            print(results[['product name', 'price']].head(3))

        else:
            print("Bot: Sorry, I didn't understand. Try asking about products 😊")

# Run chatbot
chatbot()
