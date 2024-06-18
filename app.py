import streamlit as st
from streamlit_option_menu import option_menu

from langchain_community.document_loaders import (
    PyPDFLoader, 
    WebBaseLoader, YoutubeLoader, 
    AsyncHtmlLoader 
)
from langchain_community.document_transformers import Html2TextTransformer

from langchain_community.vectorstores import Chroma, FAISS 
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.chains.summarize import load_summarize_chain
from langchain.tools.retriever import create_retriever_tool

from models import llms, embeddings
from ragchain import get_ragchain
from dfchain import DataFrameToolChain
from urlchain import get_ragagent, websearch_chain
from sqlchain import init_database, get_sqlchain

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import time
import os 
import tempfile
import shutil
import requests
import validators
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()

# Load default llm and embedding models
llm_key = "gemini-pro"
llm = llms[llm_key]
embedding = embeddings[llm_key]

# Memory is general to all chat modes
memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
loaded_memory = RunnablePassthrough.assign(
    chat_history = RunnableLambda(memory.load_memory_variables) | itemgetter("history") # Dictionary with history key after loading memory
)

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

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if chain is not None:
            with st.spinner("Thinking..."):
                question = {"question": prompt}
            
                # Format answers based on chat mode
                if selected == "PDF":
                    result = chain.invoke(question)
                    answer = result["answer"].content
                    sources = result["docs"]
                    memory.save_context(question, {"answer": answer}) 
                    print(result)

                    content = "\n\n" + "**Relevant Sources:**\n"
                    for i, doc in enumerate(sources):
                        file_name = os.path.basename(doc.metadata['source'])
                        content += f"- Source {i+1}: {file_name} (Page {doc.metadata['page']})\n"
                    complete_response = answer + content
                        
                    st.write_stream(stream_response(answer)) 
                    st.markdown(content)
                    chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": complete_response}]

                if selected == "CSV":
                    question["chat_history"] = memory.load_memory_variables({})
                    result = chain.invoke(question)

                    complete_response = ""
                    for tool_output in result:
                        output = tool_output["output"]
                        complete_response += str(output) + "\n"

                    st.markdown(complete_response)
                    chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": complete_response}]
                    memory.save_context(question, {"answer": complete_response}) 

                if selected == "SQL":
                    pass
                
                if selected == "Webpage":
                    if chain[1] == 1:
                        result = chain[0].invoke(question)
                        complete_response = ""
                        summmary = "**Summary:**  \n" + result["summmary"]
                        answer = result["answer"]
                        items = answer.strip('[]').split("], [")

                        info = []
                        for item in items:
                            # Split the item into snippet, title, and link parts
                            snippet, _, rest = item.partition(", title: ") # _ is ", title: " -> splits substrings before and after 
                            title, _, link = rest.partition(", link: ")
                            
                            # Strip the leading identifiers and whitespace from snippet, title, and link
                            result_dict = {
                                "snippet": snippet.replace("snippet: ", "").strip(),
                                "title": title.strip(),
                                "link": link.strip()
                            }
                            info.append(result_dict)

                        for item in info:
                            title = item["title"]
                            link = item["link"]
                            description = item["snippet"]
                            complete_response += f"**Title:** {title}  \n**Link:** {link}  \n**Description:** {description}\n\n"

                        st.markdown(complete_response)
                        st.write_stream(stream_response(summmary)) 
                        complete_response += summmary

                    elif chain[1] == 2:
                        result = chain[0].invoke({"input": prompt})
                        print(result)
                        complete_response = result["output"]
                        st.write_stream(stream_response(complete_response)) 

                    chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": complete_response}]

                if selected == "Youtube":
                    question["chat_history"] = memory.load_memory_variables({})
                    result = chain.invoke(question)
                    print(result)
                    st.markdown(result)
                    # chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": result}]
        else:
            st.write_stream(stream_response("Please upload your documents.")) 
    
    st.session_state.chat_histories[selected] = chat_history
    # print(f"Chat history [{selected}]: ", chat_history) # st.session_state..chat_history

def pdf_loader(docs):
    merge_docs = []
    # for file in docs:
    #     temp_dir = tempfile.mkdtemp() # Create temporary directory 
    #     temp_file_path = os.path.join(temp_dir, file.name) # Add file name in directory

    #     # Write file content in temp file
    #     with open(temp_file_path, 'wb') as temp_file:
    #         temp_file.write(file.getvalue())

    #     # Load and split content from PDF files
    #     loader = PyPDFLoader(temp_file_path)  
    #     documents = loader.load_and_split()
    #     documents = documents[:3] 
    #     merge_docs.extend(documents) # Combine list of files 
    #     shutil.rmtree(temp_dir) #Delete temporary directory 

    # index_name = "faiss_index"
    # try:
    #     vectorstore = FAISS.load_local(index_name, embedding)
    #     update_vectorstore = FAISS.from_documents(merge_docs, embedding)
    #     vectorstore.merge_from(update_vectorstore)
    #     print("Existing vectorstore loaded...")
    # except Exception as e:
    #     print("No existing vectorstore found, creating a new one...")
    #     vectorstore = FAISS.from_documents(merge_docs, embedding)

    # print(f"Vectorstore saved with {vectorstore.index.ntotal} total entries.")
    # retreiver = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
    # vectorstore.save_local(index_name)
    # vectorstore.delete([db.index_to_docstore_id[0]])

    # vectorstore = Chroma.from_documents(merge_docs, embedding)
    # save_vectorstore = Chroma.from_documents(merge_docs, embedding, persist_directory="./chroma_db")
    load_vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    retriever = load_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})

    return retriever

st.set_page_config(page_title="Document Chatbot", page_icon="✨")
st.title("Document Chatbot 📚🤖")
# st_placeholder = st.empty()

with st.sidebar:
    url = "https://github.com/cybersamurai2410/Document-Chatbot/blob/main/README.md"
    st.markdown("Made by **Aditya.S** 🌟")
    st.write("Read documentation [here](%s)" % url)

    llm_option = st.selectbox(
        "Select your preferred LLM:",
        options=list(llms.keys()),
        index=0
    )
    st.markdown(f"**Model:** {llm_option}")
    llm = llms[llm_option]

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
            seen = set(st.session_state.processed_files[selected])
            docs = []
            for file in pdf_files:
                if file.name not in seen:
                    seen.add(file.name)
                    docs.append(file)

        try: 
            if st.button("Process"):
                with st.spinner("Processing"):
                    if docs:
                        print(docs)
                        print("Loading PDF...")
                        retriever = pdf_loader(docs) 
                        print("Retrieving chain...")
                        st.session_state.processed_files[selected].extend([doc.name for doc in docs])
                        st.session_state.chains[selected] = get_ragchain(loaded_memory, retriever, llm, st.session_state.processed_files[selected])
                
                        success = st.success("Files processed successfully")
                        time.sleep(1)
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

            seen = set(st.session_state.processed_files[selected])
            docs = []
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
                            st.session_state.processed_files[selected].append(doc)

                        dfchain = DataFrameToolChain(dataframes, llm)
                        st.session_state.chains[selected] = dfchain.get_dfchain()
                
                if docs:
                    success = st.success("Files processed successfully")
                    time.sleep(1)
                    success.empty()

        except Exception as e:
            error = st.error(f"Error processing files:\n {str(e)}")

        # print(f"{selected}: {st.session_state.processed_files[selected]}")
        if st.session_state.processed_files[selected]:
            st.markdown("**Processed Files:**")
            for doc in st.session_state.processed_files[selected]:
                print(doc)
                st.write('- ', doc.name)
                doc.seek(0)
                df = pd.read_csv(doc)
                st.write(df.head())

                st.markdown("**Data Visualisation:**")
                plot_type = st.radio("Choose the type of plot", ["Line", "Bar"], key=f"plot_type_{doc.name}")

                if plot_type == 'Line':
                    x_column = st.selectbox("Choose the X-axis column", df.columns, key=f"x_col_{doc.name}")
                    y_column = st.selectbox("Choose the Y-axis column", df.columns, key=f"y_col_{doc.name}")
                    if st.button("Generate Line Plot", key=f"line_{doc.name}"):
                        # st.line_chart(df[[x_column, y_column]])
                        fig, ax = plt.subplots()
                        ax.plot(df[x_column], df[y_column])
                        ax.set_xlabel(x_column)
                        ax.set_ylabel(y_column)
                        ax.set_title(f"Line Plot of {y_column} vs {x_column}")
                        st.pyplot(fig)

                elif plot_type == 'Bar':
                    x_column = st.selectbox("Choose the X-axis column", df.columns, key=f"x_col_{doc.name}")
                    y_column = st.selectbox("Choose the Y-axis column", df.columns, key=f"y_col_{doc.name}")
                    if st.button("Generate Bar Chart", key=f"bar_{doc.name}"):
                        fig, ax = plt.subplots()
                        ax.bar(df[x_column], df[y_column])
                        ax.set_xlabel(x_column)
                        ax.set_ylabel(y_column)
                        ax.set_title(f"Bar Chart of {y_column} vs {x_column}") 

                        st.pyplot(fig)

    if selected == "SQL":
        st.title(f"Chat with {selected}") 

        st.subheader("Settings")
        st.text_input("Host", value="localhost", key="Host")
        st.text_input("Port", value="3306", key="Port")
        st.text_input("User", key="User")
        st.text_input("Password", type="password", key="Password") 
        st.text_input("Database", value="Chinook", key="Database")

        if st.button("Connect"):
            with st.spinner("Connecting to database..."):
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                print(st.session_state.db)
                if st.session_state.db is not None:
                    st.success("Connected to database")
                else:
                    st.error("Database connection failed")

                st.session_state.chains[selected] = get_sqlchain(db, llm) 

    if selected == "Webpage":
        st.title(f"Chat with {selected}")
        on = st.toggle("Toggle to manually enter URL")

        if on:
            url = st.text_input("Enter URL: ")

            if 'urls' not in st.session_state:
                st.session_state.urls = []

            if 'url_groups' not in st.session_state:
                st.session_state.url_groups = {}

            if st.button("Add URL"):
                if url not in st.session_state.urls and validators.url(url):
                    st.session_state.urls.append(url)
                    
                    success = st.success("URL added")
                    time.sleep(1)
                    success.empty()
                else:
                    st.error("URL is already added or is invalid")
            
            if st.session_state.urls:
                st.write("URLs List:")
                for u in st.session_state.urls:
                    st.write('-', u)

            if st.button("Process"):
                with st.spinner("Processing"):
                    try:
                        loader = WebBaseLoader(st.session_state.urls)
                        docs = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        doc_splits = text_splitter.split_documents(docs)

                        vectorstore = FAISS.load_local("webpage_index", embedding, allow_dangerous_deserialization=True)
                        # vectorstore = FAISS.from_documents(doc_splits, embedding)
                        # vectorstore.save_local("webpage_index")
                        retriever = vectorstore.as_retriever()

                        st.session_state.chains[selected] = (get_ragagent(llm, retriever), 2)

                        success = st.success("URLs processed successfully")
                        time.sleep(1)
                        success.empty()

                    except Exception as e:
                        error = st.error(f"Error processing URLs:\n {str(e)}")
        else:
            st.markdown("✨ *Enter your prompt and query based on knowledge retrieved from websearch.*")
            st.session_state.chains[selected] = (websearch_chain(llm), 1)

    if selected == "YouTube": 
        st.title(f"Chat with {selected}")
        url = st.text_input("Enter URL: ")
        if st.button("Process"):
            with st.spinner("Processing"):
                try:
                    st.write(f"URL: {url}")
                    st.video(url, subtitles="subtitles.vtt") # Display youtube video

                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    doc_splits = text_splitter.split_documents(docs)

                    index_name = "doc-chatbot"
                    vectorstore = PineconeVectorStore.from_documents(
                        doc_splits, 
                        embedding, 
                        index_name=index_name,
                        # namespace="example-namespace"
                    )
                    # PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
                    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
                    # vectorstore.add_documents(doc_splits, namespace="example-namespace") # Partition the records in an index into namespaces
                    # vectorstore.delete([0])

                    # Generate summary  
                    summary_chain = load_summarize_chain(llm, chain_type="map-reduce")
                    summary = summary_chain.invoke(docs)
                    st.markdown(f"*Summary:* \n{summary}")

                except Exception as e:
                            error = st.error(f"Error processing URL:\n {str(e)}")

# Process user prompt 
if prompt := st.chat_input("Ask anything..."):
    chat(prompt, selected) 

# streamlit run app.py
# C:\Users\Dell\AppData\Local\Temp\tmpwc1bhv2v.pdf
# https://python.langchain.com/docs/modules/chains/
    