import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get API key
groq_api_key = st.secrets["GROQ_API_KEY"]

# ⚠️ Replace with a currently supported model
model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser = StrOutputParser()

# LangChain chain
chain = prompt_template | model | parser

# Streamlit UI
st.set_page_config(page_title="Groq Translation App")
st.title("🌍 LangChain Translation using Groq")

# Input fields
text = st.text_area("Enter the text to translate", "Hello, how are you?")
language = st.selectbox("Select target language", ["French", "Spanish", "German", "Hindi", "Tamil", "Arabic","Urdu","Telugu"])

# Button to run
if st.button("Translate"):
    if text and language:
        with st.spinner("Translating..."):
            try:
                result = chain.invoke({
                    "text": text,
                    "language": language
                })
                st.success("✅ Translation complete!")
                st.text_area("Translated Text", result, height=150)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter both text and language.")
