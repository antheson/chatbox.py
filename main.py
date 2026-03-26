import pandas as pd
import random
import os

# Debug: check files
print("Files in directory:", os.listdir())

# Load dataset
df = pd.read_csv("diversified_ecommerce_dataset.csv")

# Clean column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

print("Columns:", df.columns)

def chatbot():
    print("🤖 ShopAssist Bot: Hello! Ask me about products 😊")

    while True:
        user_input = input("You: ").lower()

        if user_input in ["bye", "exit"]:
            print("Bot: Goodbye!")
            break

        elif "hello" in user_input or "hi" in user_input:
            print("Bot: Hi! How can I help you today?")

        # Cheap product
        elif "cheap" in user_input:
            cheap_products = df[df['price'] < df['price'].mean()]
            result = cheap_products.sample(1)

            print("Bot: I recommend this budget product:")
            print(result[['product_name', 'brand', 'price']])

        # Best rated
        elif "best" in user_input or "top" in user_input:
            if 'rating' in df.columns:
                best = df.sort_values(by='rating', ascending=False).head(1)
                print("Bot: Best rated product:")
                print(best[['product_name', 'brand', 'rating']])
            else:
                print("Bot: Rating data not available.")

        # Category search
        elif "lipstick" in user_input:
            results = df[df['subcategory'].str.contains("lipstick", case=False, na=False)]
            print("Bot: Here are some lipsticks:")
            print(results[['product_name', 'brand', 'price']].head(3))

        # Brand search
        elif "brand" in user_input:
            brand_name = user_input.split("brand")[-1].strip()
            results = df[df['brand'].str.contains(brand_name, case=False, na=False)]

            if not results.empty:
                print(f"Bot: Products from {brand_name}:")
                print(results[['product_name', 'price']].head(3))
            else:
                print("Bot: No products found for that brand.")

        else:
            print("Bot: Sorry, I didn't understand. Try asking about products 😊")

chatbot()
