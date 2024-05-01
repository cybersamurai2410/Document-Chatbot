from operator import itemgetter

from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import format_document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable



def get_urlchain(loaded_memory, llm, retriever):
    
    template = """"""

def websearch_chain(llm):
    '''
    1. Convert prompt to search query.
    2. Call search tool to retreive top k results.
    3. Return summaries of search results with sources. 
    '''

    template = """Convert the following prompt to a query for web search: {question}\n
    Search Query: """
    PROMPT_TO_SEARCH = PromptTemplate.from_template(template)

    generate_searchquery = {"search_query": RunnablePassthrough() | PROMPT_TO_SEARCH | llm | StrOutputParser}

    chain  = generate_searchquery

    return chain 