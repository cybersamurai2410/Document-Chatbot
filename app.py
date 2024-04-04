import streamlit as st
from streamlit_option_menu import option_menu

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.tools.retriever import create_retriever_tool

from models import llms, embeddings
from ragchain import combine_documents, CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT
from dotenv import load_dotenv
from operator import itemgetter
import time
import os 
from io import BytesIO
import tempfile
import shutil

load_dotenv()
llm = llms['gemini-pro']
embedding = embeddings["gemini-pro"]

memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
loaded_memory = RunnablePassthrough.assign(
    chat_history = RunnableLambda(memory.load_memory_variables) | itemgetter("history") # Dictionary with history key after loading memory
)

def chat(prompt, docs=[], chain=None):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def stream_response(message):
        for word in message.split():
            yield word + " "
            time.sleep(0.05)

    if chain is not None:
        result = chain.invoke({"question": prompt})
        print(result)
        memory.save_context(prompt, {"answer": result["answer"].content}) 

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if docs:
            response = st.write_stream(stream_response(result["answer"].content)) 
            st.session_state.chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        else:
            st.write_stream(stream_response("Please upload your documents."))
    
    print("Chat history: ", st.session_state.chat_history)

def pdf_loader(docs):
    merge_docs = []
    for file in docs:
        temp_dir = tempfile.mkdtemp() # Create temporary directory 
        temp_file_path = os.path.join(temp_dir, file.name) # Add file name in directory

        # Write file content in temp file
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.getvalue())

        # Load and split content from PDF files
        loader = PyPDFLoader(temp_file_path)  
        documents = loader.load_and_split()
        documents = documents[:3] 
        merge_docs.extend(documents) # Combine list of files 
        shutil.rmtree(temp_dir) #Delete temporary directory 

    print(merge_docs)
    vectorstore = Chroma.from_documents(merge_docs, embedding)
    # save_vectorstore = Chroma.from_documents(documents, embedding, persist_directory="./chroma_db")
    # load_vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    retriever = vectorstore.as_retriever()

    return retriever

def get_ragchain(retriever):
    standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]) # Combine chat history as single string
    }
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser()
}

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever, # Retrieve list of sources 
        "question": lambda x: x["standalone_question"]
    }

    answer = {
        "answer": {
            "context": lambda x: combine_documents(x["docs"]), # x is dictionary from retrieved_documents with key passed as parameter to return context
            "question": itemgetter("question") # Gets question key from retrieved_documents dictionary 
            } | ANSWER_PROMPT | llm,
        "docs": itemgetter("docs")
    }

    chain = loaded_memory | standalone_question | retrieved_documents | answer

    return chain

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
                    # print(docs)
                    retriever = pdf_loader(docs)
                    rag_chain = get_ragchain(retriever)
                
                success = st.success("Files processed successfully")
                time.sleep(3)
                success.empty()

        except Exception as e:
            error = st.error(f"Error uploading files:\n {str(e)}")

st.title("Document Chatbot 📚🤖")
if prompt := st.chat_input("Ask anything..."):
    chat(prompt, docs, rag_chain)

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
    