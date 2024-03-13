from dotenv import load_dotenv
import streamlit as st 
from PyPDF2 import PdfReader

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-pro")

def extract_document(docs):
    pass

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with your documents", page_icon=":books:")

    st.header("Chat with your documents :books:")
    st.text_input("Ask a question:")

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract document text
                raw_text = extract_document(docs) 

                # Retrieve text chunks

                # Create vector store

if __name__ == '__main__':
    main()

'''
streamlit run app.py
'''
