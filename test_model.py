import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def test_model():
    print("Initializing ChatGroq with meta-llama/llama-4-scout-17b-16e-instruct...")
    try:
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0,
            max_retries=2
        )
        print("Sending a test prompt: 'Hello, what is 2+2?'")
        response = llm.invoke("Hello, what is 2+2? Please keep your answer under 10 words.")
        print("\n--- Response ---")
        print(response.content)
        print("----------------\n")
        print("✅ The model is working perfectly!")
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    test_model()