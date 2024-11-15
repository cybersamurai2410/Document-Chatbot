import streamlit as st
from streamlit_option_menu import option_menu

from langchain_community.document_loaders import (
    PyPDFLoader, 
    WebBaseLoader, 
    YoutubeLoader, 
)
from langchain_community.document_loaders.youtube import TranscriptFormat

from langchain_community.vectorstores import Chroma 
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.chains.summarize import load_summarize_chain
from pinecone import Pinecone, ServerlessSpec

from models import llms, embeddings
from ragchain import get_ragchain
from dfchain import DataFrameToolChain
from urlchain import get_ragagent, websearch_chain, youtube_chain
from sqlchain import init_database, get_sqlchain

from uuid import uuid4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import time
import os 
import tempfile
import shutil
import validators
from bs4 import BeautifulSoup
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Load default llm and embedding models
default_model = "gpt-4o-mini (openai)"
llm = llms[default_model]
embedding = embeddings["openai_gpt"] 

# Memory is general to all chat modes
memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
loaded_memory = RunnablePassthrough.assign(
    chat_history = RunnableLambda(memory.load_memory_variables) | itemgetter("history") # Dictionary with history key after loading memory: loaded_memory.invoke({})
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
                    print(result)

                    answer = result["answer"].content
                    sources = result["docs"]
                    memory.save_context(question, {"answer": answer}) 

                    content = "\n\n" + "**Relevant Sources:**\n"
                    for i, (doc, score) in enumerate(sources):
                        file_name = os.path.basename(doc.metadata['source'])
                        content += f"- Source {i+1}: {file_name} | Page {doc.metadata['page']} (Confidence Score: {score*100:.2f}%) \n"
                    complete_response = answer + content
                        
                    st.write_stream(stream_response(answer)) 
                    st.markdown(content)
                    chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": complete_response}]

                if selected == "CSV":
                    question["chat_history"] = memory.load_memory_variables({})["history"] # Variable chat_history should be a list of base messages 
                    current_time = datetime.now()
                    result = chain.invoke(question)
                    print("Result: ", result)

                    complete_response = ""
                    for tool_output in result:
                        # output = tool_output["output"]
                        output = tool_output.get("explanation", None)  # If key does not exist then None 
                        complete_response += str(output.content) + "\n" if output is not None else ""

                    # Get list of recently generated graphs 
                    print("Directory: ", os.listdir('Graphs'))
                    files_after_current_time = [file for file in os.listdir('Graphs') if os.path.getmtime(os.path.join('Graphs', file)) > current_time.timestamp()]
                    print("Files: ", files_after_current_time)
                    
                    # Display generated graphs
                    if files_after_current_time:
                        complete_response += "\n\n**Generated Graphs:**\n\n"
                        for file in files_after_current_time:
                            complete_response += f"![{file}](Graphs/{file})\n\n"
                            st.image(f"Graphs/{file}", caption=file)
                    
                    print("Response: ", complete_response)
                    st.markdown(complete_response)
                    chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": complete_response}]
                    memory.save_context(question, {"answer": complete_response}) 

                if selected == "SQL":
                    question["chat_history"] = memory.load_memory_variables({})["history"]
                    result = chain.invoke(question)
                    print("Result: ", result)

                    answer = result["answer"]
                    answer += f"\n\n **Query:** \n\n```sql {result['query']}```"
                    st.markdown(answer)

                    chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                    memory.save_context(question, {"answer": answer}) 
                
                if selected == "Webpage":
                    if chain[1] == 1:
                        result = chain[0].invoke(question)
                        print(result)

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
                        result = chain[0].invoke({"input": prompt, "chat_history": memory.load_memory_variables({})["history"], "agent_scratchpad": ""})
                        print(result)

                        complete_response = result["output"]
                        st.write_stream(stream_response(complete_response)) 

                    memory.save_context(question, {"answer": complete_response}) 
                    chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": complete_response}]

                if selected == "YouTube":
                    print("Youtube chain...")
                    result = chain.invoke({"input": prompt, "chat_history": memory.load_memory_variables({})["history"]})
                    print("Result: ", result)
                    st.markdown(result)

                    memory.save_context(question, {"answer": complete_response}) 
                    chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": result}]
        else:
            st.write_stream(stream_response("Please upload your documents.")) 
    
    st.session_state.chat_histories[selected] = chat_history

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
        documents = documents[:3] # Take first three pages of the file (proof of concept)
        merge_docs.extend(documents) # Combine list of files 
        shutil.rmtree(temp_dir) # Delete temporary directory
        print(documents, "\n") 

    vector_store = Chroma(
        collection_name="pdf",
        embedding_function=embedding,
        persist_directory="./documents-db",   
    )
    vector_store.add_documents(merge_docs)
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.5})

    return vector_store, retriever

def init_vector_db(index_name):
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    # existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    # if index_name not in existing_indexes:
    #     pc.create_index(
    #         name=index_name,
    #         dimension=1536, # openai embedding dimensions 
    #         metric="cosine",
    #         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    #     )
    #     while not pc.describe_index(index_name).status["ready"]:
    #         time.sleep(1)

    index = pc.Index(index_name)

    return index 

st.set_page_config(page_title="Document Chatbot", page_icon="✨")
st.title("Document Chatbot 📚🤖")

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
                        vectorstore, retriever = pdf_loader(docs) 
                        print("Retrieving chain...")
                        st.session_state.processed_files[selected].extend([doc.name for doc in docs])
                        st.session_state.chains[selected] = get_ragchain(loaded_memory, vectorstore, retriever, llm, st.session_state.processed_files[selected])
                
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

            seen = set(file.name for file in st.session_state.processed_files[selected]) # Hashable attribute of uploaded csv files 
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
                            filename = doc.name.split('.')[0]
                            dataframes[filename] = df
                            st.session_state.processed_files[selected].append(doc)

                        print(dataframes)
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

                        vector_store = Chroma(
                            collection_name="webpages",
                            embedding_function=embedding,
                            persist_directory="./documents-db",   
                        )
                        # vector_store.add_documents(doc_splits)
                        retriever = vector_store.as_retriever()

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
        name_id = st.text_input("Enter a name for the video: ")
        name_id_format = name_id.lower().replace(" ", "_").replace("-", "_") # Format the name to add as prefix to id for each embedding

        st.session_state.video_url = st.session_state.get('video_url', "")
        st.session_state.name_id = st.session_state.get('name_id', "")
        if 'summary' not in st.session_state:
            st.session_state.summary = ""

        if st.button("Process"):
            with st.spinner("Processing"):
                try:
                    if url and validators.url(url):
                        st.session_state.video_url = url
                    if name_id:
                        st.session_state.name_id = name_id

                    # Chunking text based on time stamps 
                    loader = YoutubeLoader.from_youtube_url(
                        url,
                        add_video_info=False,
                        transcript_format=TranscriptFormat.CHUNKS,
                        chunk_size_seconds=30,
                    )

                    if name_id != "":
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["video_name"] = name_id 
                        # print(docs)

                        # uuids = [f"{name_id_format}-{str(uuid4())}" for _ in range(len(docs))]
                        index_name = "doc-chatbot-vectordb"
                        index = init_vector_db(index_name)
                        vector_store = PineconeVectorStore(index=index, embedding=embedding, namespace=name_id)
                        # vector_store.add_documents(documents=docs, ids=uuids)
                        retriever = vector_store.as_retriever()
                        # vector_store.delete(ids=[uuids[-1]], namespace=name_id) # delete_all=True to clear index  

                        st.session_state.chains[selected] = youtube_chain(llm, retriever)
                    else:
                        st.error("Video name not provided.")

                    # Generate summary 
                    with st.spinner("Generating Summary"): 
                        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
                        summary = summary_chain.invoke(docs)
                        st.session_state.summary = summary['output_text']
                        success = st.success("Summary Generated")

                except Exception as e:
                    error = st.error(f"Error processing URL:\n {str(e)}") 
                
        if st.session_state.video_url:
            st.markdown(f"**Link:** [{st.session_state.name_id}]({st.session_state.video_url})")
            st.video(st.session_state.video_url) # Display youtube video
        
        if st.session_state.summary:
            st.markdown(f"**Summary:** \n\n{st.session_state.summary}")

# Process user prompt 
if prompt := st.chat_input("Ask your question..."):
    chat(prompt, selected) 

# streamlit run app.py
    