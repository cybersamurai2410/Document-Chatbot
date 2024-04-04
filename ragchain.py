from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import format_document

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
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
