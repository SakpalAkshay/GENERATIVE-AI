from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

print("Started")


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
result = model.invoke("Tell me a poem of 5 lines on Football")
print(result.content)

