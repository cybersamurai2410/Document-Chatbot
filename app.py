import streamlit as st
from streamlit_option_menu import option_menu

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from langchain.output_parsers import ResponseSchema, StructuredOutputParser
# from langchain.tools.retriever import create_retriever_tool
from langchain.chains.summarize import load_summarize_chain

from models import llms, embeddings
from ragchain import get_ragchain
from dfchain import get_dfchain

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from operator import itemgetter
import time
import os 
from io import BytesIO, StringIO
import tempfile
import shutil

load_dotenv()
llm = llms['gemini-pro']
embedding = embeddings["gemini-pro"]

# summary_chain = load_summarize_chain(llm, chain_type="map-reduce")
# summary = summary_chain.invoke(merge_docs)

memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
loaded_memory = RunnablePassthrough.assign(
    chat_history = RunnableLambda(memory.load_memory_variables) | itemgetter("history") # Dictionary with history key after loading memory
)
# Ensure memory is seperate for each mode

# Store conversation for each mode
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {
        "PDF": [],
        "CSV": [],
        "SQL": [],
        "Webpage": [],
        "YouTube": [],
    }

# Store LLM chains 
if 'chains' not in st.session_state:
    st.session_state.chains = {
        'PDF': None,
        'CSV': None,
        "SQL": None,
        "Webpage": None,
        "YouTube": None,
    }

# Store list of processed files 
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {
        'PDF': [],
        'CSV': [],
    }

def chat(prompt, selected):
    chat_history = st.session_state.chat_histories[selected]
    chain = st.session_state.chains[selected]

    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def stream_response(message):
        for word in message.split():
            yield word + " "
            time.sleep(0.05)

    if chain is not None:
        with st.spinner("Processing"):
            question = {"question": prompt}
            result = chain.invoke(question)
            # print(result)
            memory.save_context(question, {"answer": result["answer"].content}) 

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if chain is not None:
            answer = result["answer"].content
            sources = result["docs"]
            
            content = "\n\n" + "**Relevant Sources:**\n"
            for i, doc in enumerate(sources):
                content += f"- Source {i+1}: {doc.metadata['source']} (Page {doc.metadata['page']})\n"
            complete_response = answer + content
                
            st.write_stream(stream_response(answer)) 
            st.markdown(content)
            chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": complete_response}]
        else:
            st.write_stream(stream_response("Please upload your documents."))
    
    st.session_state.chat_histories[selected] = chat_history
    print(f"Chat history [{selected}]: ", chat_history) # st.session_state..chat_history

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

    # print(merge_docs)
    # vectorstore = Chroma.from_documents(merge_docs, embedding)

    # vectorstore_directory = "./"+file.name
    # save_vectorstore = Chroma.from_documents(merge_docs, embedding, persist_directory="./chroma_db")
    load_vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

    retriever = load_vectorstore.as_retriever()

    return retriever

st.set_page_config(page_title="Document Chatbot", page_icon="✨")
st.title("Document Chatbot 📚🤖")
# st_placeholder = st.empty()

with st.sidebar:
    url = "https://github.com/cybersamurai2410/Document-Chatbot/blob/main/README.md"
    st.markdown("Made by **Aditya.S** 🌟")
    st.write("Read documentation [here](%s)" % url)

    selected = option_menu(
        menu_title="Main Menu",
        options=["PDF", "CSV", "SQL", "Webpage", "YouTube"],
        icons=["filetype-pdf", "filetype-csv", "filetype-sql", "link", "youtube"], # https://icons.getbootstrap.com/
        default_index=0,
    )

    if selected == "PDF":
        st.title(f"Chat with {selected}")
        uploaded_files = st.file_uploader(f"Upload your {selected} files here and click on **Process**", accept_multiple_files=True, key="pdf_uploader")

        if uploaded_files is not None:
            pdf_files = [file for file in uploaded_files if file.type == "application/pdf"] # Filter files to PDF 

            # Remove duplicate files
            seen = set()
            docs = list()
            for file in pdf_files:
                if file.name not in seen:
                    seen.add(file.name)
                    docs.append(file)
        # uploaded_files = None

        try: 
            if st.button("Process"):
                with st.spinner("Processing"):
                    if docs:
                        print("Loading PDF...")
                        retriever = pdf_loader(docs)
                        print("Retrieving chain...")
                        st.session_state.chains[selected] = get_ragchain(loaded_memory, retriever, llm)

                        st.session_state.processed_files[selected] = [doc.name for doc in docs]
                
                if docs:
                    success = st.success("Files processed successfully")
                    time.sleep(3)
                    success.empty()

        except Exception as e:
            error = st.error(f"Error processing files:\n {str(e)}")

        print(f"{selected}: {st.session_state.processed_files[selected]}")
        if st.session_state.processed_files[selected]:
            st.markdown("**Processed Files:**")
            for file_name in st.session_state.processed_files[selected]:
                st.write('- ', file_name) 

    if selected == "CSV":
        st.title(f"Chat with {selected}")
        uploaded_files = st.file_uploader(f"Upload your {selected} files here and click on **Process**", accept_multiple_files=True, key="csv_uploader")

        if uploaded_files is not None:
            csv_files = [file for file in uploaded_files if file.type == "text/csv"] # Filter files to CSV 

            seen = set()
            docs = list()
            for file in csv_files:
                if file.name not in seen:
                    seen.add(file.name)
                    docs.append(file)
        
        try:
            if st.button("Process"):
                with st.spinner("Processing"):

                    if docs:
                        dataframes = {}
                        st.session_state.processed_files[selected] = []

                        for doc in docs:
                            df = pd.read_csv(doc)
                            dataframes[doc.name] = df
                            st.session_state.processed_files[selected].append(doc.name)

                        # print(docs)
                        # get_dfchain(dataframes)
                
                if docs:
                    success = st.success("Files processed successfully")
                    time.sleep(3)
                    success.empty()

        except Exception as e:
            error = st.error(f"Error processing files:\n {str(e)}")

        print(f"{selected}: {st.session_state.processed_files[selected]}")
        if st.session_state.processed_files[selected]:
            st.markdown("**Processed Files:**")
            for doc in st.session_state.processed_files[selected]:
                st.write('- ', doc.name)
                doc.seek(0)
                df = pd.read_csv(doc)
                st.write(df.head())

    if selected == "SQL":
        st.title(f"Chat with {selected}")
    if selected == "Webpage":
        st.title(f"Chat with {selected}")
    if selected == "YouTube":
        st.title(f"Chat with {selected}")

# Process user prompt 
if prompt := st.chat_input("Ask anything..."):
    chat(prompt, selected) 

# streamlit run prototype.py
# C:\Users\Dell\AppData\Local\Temp\tmpwc1bhv2v.pdf
    