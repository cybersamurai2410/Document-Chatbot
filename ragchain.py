from operator import itemgetter

from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import format_document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    # Fromat documents to add as context 
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# input_variables=['chat_history', 'question']
template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question if it is relevant to the chat history or else just return the original question.\n
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""" 
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

template = """Answer the question in a few sentences based on the following context if it is relevant or else use your own knowledge to answer the question and mention that the context was not relevant.

Context:
{context}

Question: {question}

These are the names of the files you have access to where context is retrieved: {files}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template) 

# RAG Chain for reading PDF files
def get_ragchain(loaded_memory, retriever, llm, files):

    standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]) # Combine chat history as single string
    }
    | CONDENSE_QUESTION_PROMPT # Passes {"question", "chat_history"}
    | llm
    | StrOutputParser()
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever, # Retrieve list of sources passing just standalone question to retriever
        "question": lambda x: x["standalone_question"]
    }

    answer = {
        "answer": {
            "context": lambda x: combine_documents(x["docs"]), # x is dictionary from retrieved_documents with key passed as parameter to return context
            "question": itemgetter("question"), # Gets question key from retrieved_documents dictionary 
            "files": RunnablePassthrough(lambda *args: files)
            } | ANSWER_PROMPT | llm,
        "docs": itemgetter("docs")
    }

    # Only runnables and dicts can be passed in pipeline
    chain = loaded_memory | standalone_question | retrieved_documents | answer
    # print(chain) 

    return chain
