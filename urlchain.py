from operator import itemgetter

from langchain.agents import AgentExecutor, AgentType, Tool, tool, create_structured_chat_agent
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain import hub
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable
from langchain.tools.retriever import create_retriever_tool

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# RAG Agent 
def get_ragagent(llm, retriever):
    
    prompt = hub.pull("hwchase17/structured-chat-agent")

    tools = []
    retriever_tool = create_retriever_tool(
    retriever = retriever,
    name = "webpages", # Group name or url list
    description = "Information from corpuses for webpages.", # Summary of webpages 
    )
    tools.append(retriever_tool) # Loop retriever for different groups of urls then add to tools list

    agent = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

def websearch_chain(llm):
    # search = GoogleSearchAPIWrapper(k=1)
    search = DuckDuckGoSearchResults()
    search_tool = Tool(
        name="web_broswer",
        description="Search web browser for recent results.",
        func=search.run,
    )

    template = """Convert the following question prompt to a single query optimized for web search: {question}\n
    Just return the search query and nothing else!
    Search Query: """
    PROMPT_TO_SEARCH = PromptTemplate.from_template(template)

    def remove_single_quotes(search_query):
        return search_query.strip('"')

    generate_searchquery = PROMPT_TO_SEARCH | llm | StrOutputParser() | remove_single_quotes
    chain = (
        RunnablePassthrough.assign(search_query=generate_searchquery)
        .assign(answer = itemgetter("search_query") | search_tool)
        .pick(["search_query", "answer"])
    )

    return chain 

def youtube_chain(llm, retreiver):
    pass

# question = "what are the results of the last ufc event?"
# chain = websearch_chain(llm)
# result = chain.invoke({"question": question})
# print(result)
"""
[snippet: From a perfect uppercut in the welterweight opener to a massive right hook that ended the evening, the first event after UFC 300 was an explosive affair that produced plenty of highlights and gave ..., title: Main Card Results | UFC Fight Night: Nicolau vs Perez, link: https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-fight-night-nicolau-vs-perez?language_content_entity=en], [snippet: See The Fight Results, Watch Post-Fight Interviews With The Main Card Winners And More From UFC 298: Volkanovski vs Topuria, Live From Honda Center In Anaheim By E. Spencer Kyte, on X @spencerkyte ..., title: Main Card Results | UFC 298: Volkanovski vs Topuria - UFC.com, link: https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-298-volkanovski-vs-topuria?language_content_entity=en], [snippet: Ultimate Fighting Championship (UFC) was back at the friendly confines of the Apex last night (Sat., April 27, 2024) in Las Vegas, Nevada for UFC Vegas 91. Headlining the event was a Flyweight ..., title: UFC Vegas 91 results: Biggest winners, loser from 'Nicolau vs. Perez ..., link: https://www.mmamania.com/2024/4/28/24143415/ufc-vegas-91-results-biggest-winners-loser-nicolau-perez-last-night-espn-mma], [snippet: Esteban Ribovics (29-28, 29-28, 29-28) defeats Kamuela Kirk by unanimous decision . Esteban Ribovics and Kamuela Kirk set the Fight of the Night bar high right out of the gate on Saturday night ..., title: UFC 290: Volkanovski vs Rodriguez Final Results - UFC.com, link: https://www.ufc.com/news/ufc-290-volkanovski-vs-rodriguez-results-highlights-winner-interviews?language_content_entity=en]
"""
