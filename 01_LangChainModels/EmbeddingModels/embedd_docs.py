from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

print("Working on Embeddings")

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]
embedding = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

result = embedding.embed_documents(documents)
print("Query Result: " + str(result))
