import streamlit as st
from streamlit_option_menu import option_menu

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from models import llms, embeddings
from dotenv import load_dotenv
import time
import os 
from io import BytesIO
import tempfile

load_dotenv()
llm = llms['gemini-pro']
embedding = embeddings["gemini-pro"]

def chat(prompt, docs):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def stream_response(message):
        for word in message.split():
            yield word + " "
            time.sleep(0.05)

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if docs:
            response = st.write_stream(stream_response(prompt)) 

            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.write_stream(stream_response("Please upload your documents."))
    
    print("Chat history: ", st.session_state.chat_history)

def document_loader(docs):
    merge_docs = []
    for file in docs:
        temp_file_descriptor, temp_file_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_file_descriptor)

        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.getvalue())

        loader = PyPDFLoader(temp_file_path)  
        documents = loader.load_and_split()
        documents = documents[:3] 
        merge_docs.extend(documents)
        os.remove(temp_file_path)

    print(merge_docs)
    vectorstore = Chroma.from_documents(documents, embedding)
    retriever = vectorstore.as_retriever()

    return retriever

st.set_page_config(page_title="Document Chatbot", page_icon="✨")

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["PDF", "CSV", "SQL", "Webpage", "YouTube"],
        icons=["filetype-pdf", "filetype-csv", "filetype-sql", "link", "youtube"], # https://icons.getbootstrap.com/
        default_index=0,
    )

    if selected == "PDF":
        st.title(f"Chat with {selected}")
        docs = st.file_uploader(f"Upload your {selected} files here and click on **Process**", accept_multiple_files=True)

        if docs is not None:
            docs = [doc for doc in docs if doc.type == "application/pdf"]
        
        # Show names of files once processed
        try: 
            if st.button("Process"):
                with st.spinner("Processing"):
                    print(docs)
                    document_loader(docs)
                
                success = st.success("Files processed successfully")
                time.sleep(3)
                success.empty()

        except Exception as e:
            error = st.error(f"Error uploading files:\n {str(e)}")

st.title("Document Chatbot 📚🤖")
if prompt := st.chat_input("Ask anything..."):
    chat(prompt, docs)

if selected == "CSV":
    st.title(f"Chat with {selected}")
if selected == "SQL":
    st.title(f"Chat with {selected}")
if selected == "Webpage":
    st.title(f"Chat with {selected}")
if selected == "YouTube":
    st.title(f"Chat with {selected}")

# streamlit run prototype.py
# C:\Users\Dell\AppData\Local\Temp\tmpwc1bhv2v.pdf
    