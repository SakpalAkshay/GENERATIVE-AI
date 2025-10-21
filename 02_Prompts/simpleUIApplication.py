from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
import streamlit as st
st.header("Simple Gemini Chat Application")
user_input = st.text_input("Please enter your query for Gemini....")

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
if st.button('Go.....'):
    result = model.invoke(user_input)

    st.write(result.content)