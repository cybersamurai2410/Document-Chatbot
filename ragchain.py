from operator import itemgetter

from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import format_document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    # Fromat documents to add as context 
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# input_variables=['chat_history', 'question']
template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question or else return the original question if there is no previous conversation.\n
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""" 
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

template = """Answer the question in a few sentences based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# RAG Chain for reading PDF files
def get_ragchain(loaded_memory, retriever, llm):
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
