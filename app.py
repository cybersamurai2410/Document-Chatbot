from dotenv import load_dotenv
import streamlit as st 
from PyPDF2 import PdfReader
from enum import Enum 

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS 

llm = ChatGoogleGenerativeAI(model="gemini-pro") # Create dictionary of LLMs  
Documents = Enum('Doc_Type', ['PDF', 'WEBPAGE', 'YOUTUBE', 'TEXT', 'CSV'])

# 1 token ~= 4 chars in English; cost = estimate_tokens * price per token
def estimate_tokens(text, token_cost):
    token_size = len(text)/4
    cost = token_size*token_cost
    return token_size, cost

#---------------------
# Document Q&A
def document_loader():
    '''
    Documents:
        - PDF (.pdf)
        - Webpage 
        - YouTube
        - Text (.txt)
        - CSV (.csv)
    '''
    pass

# SQL Q&A
def database_loader():
    pass

def image_loader():
    pass

#---------------------

def extract_document(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len 
    )
    chunks = text_splitter.split_text(text)

    return chunks

def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore

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
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

if __name__ == '__main__':
    main()

'''
streamlit run app.py
'''
