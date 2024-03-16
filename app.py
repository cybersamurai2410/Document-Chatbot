from dotenv import load_dotenv
import time
import random 
import streamlit as st 
from PyPDF2 import PdfReader
from enum import Enum 

# LLMs and Embeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain, LLMChain

llm = {
    "gemini-pro": ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    }  
Documents = Enum('Doc_Type', ['PDF', 'WEBPAGE', 'YOUTUBE', 'TEXT', 'CSV'])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 1 token ~= 4 chars in English; cost = estimate_tokens * price per token
def estimate_tokens(text, token_cost):
    token_size = len(text)/4
    cost = token_size*token_cost
    return token_size, cost

#---------------------
# Document Q&A
def document_loader(doc_type=0):
    '''
    Documents:
        - PDF (.pdf)
        - Webpage 
        - YouTube
        - Text (.txt)
        - CSV (.csv) -> integrate agents for data science with dataframe
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

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm['gemini-pro'],
        retriever=vectorstore.as_retriever(),
        memory=memory 
    )

    return conversation_chain

def get_response(user_prompt):
    response = st.session_state.conversation({"question": user_prompt}) # after executing conversation = get_conversation_chain; question, chat_history and answer key from output 
    st.session_state.chat_history = response['chat_history']

    print(response, '\n')
    '''
    {'question': 'what is this document about?', 'chat_history': [HumanMessage(content='what is this document about?'), 
    AIMessage(content='I do not know the answer to this question. The provided text does not contain any information about what the document is about.')], 
    'answer': 'I do not know the answer to this question. The provided text does not contain any information about what the document is about.'}
    '''

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0: # Even numbers are user messages else AI messages
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.write_stream(stream_response(message.content))

def stream_response(message):
    for word in message.split():
        yield word + " "
        time.sleep(0.05)


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with your documents", page_icon=":books:")

    # st.session_state prevents variables from reinitializing since streamlit often reloads code during session
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your documents :books:")
    # st.text_input("Ask a question:")
    user_prompt = st.chat_input("Ask a question about your documents")
    if user_prompt:
        get_response(user_prompt)

    # # Display chat messages from history on app rerun
    # for message in st.session_state.chat_history:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # if prompt := st.chat_input("Ask a question about your documents"):
    #     st.session_state.chat_history.append({"role": "user", "content": prompt}) # Add user message to chat history

    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #     with st.chat_message("assistant"):
    #         response = st.write_stream(stream_response())
    #     st.session_state.chat_history.append({"role": "assistant", "content": response}) # Add assistant response to chat history

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = extract_document(docs) # Extract document text
                text_chunks = get_text_chunks(raw_text) # Retrieve text chunks
                print(text_chunks)
                vectorstore = get_vectorstore(text_chunks) # Create vector store
                print(vectorstore)
                st.session_state.conversation = get_conversation_chain(vectorstore) # Create conversation chain

if __name__ == '__main__':
    main()


# streamlit run app.py
# https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
