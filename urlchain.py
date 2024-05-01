from operator import itemgetter

from langchain_community.tools import DuckDuckGoSearchResults

from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import format_document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable

from dotenv import load_dotenv
load_dotenv()

from models import llms
llm = llms["gemini-pro"]


def get_urlchain(loaded_memory, llm, retriever):
    
    template = """"""

def websearch_chain(llm):

    # search = GoogleSearchAPIWrapper(k=1)
    # tool = Tool(
    #     name="web_broswer",
    #     description="Search web browser for recent results.",
    #     func=search.run,
    # )

    search = DuckDuckGoSearchResults()

    template = """Convert the following question prompt to a query optimized for web search: {question}\n
    Search Query: """
    PROMPT_TO_SEARCH = PromptTemplate.from_template(template)

    generate_searchquery = PROMPT_TO_SEARCH | llm | StrOutputParser
    chain  = generate_searchquery | search

    return chain 

# question = "what are the results of the last ufc event?"
# chain = websearch_chain(llm)
# result = chain.invoke({"question": question})
