from operator import itemgetter
import os 

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

# search = GoogleSearchAPIWrapper(k=1)
search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="web_broswer",
    description="Search web browser for recent results.",
    func=search.run,
)

# RAG Agent 
def get_ragagent(llm, retriever):

    # prompt = hub.pull("hwchase17/structured-chat-agent") # https://smith.langchain.com/hub/hwchase17/structured-chat-agent
    system = '''Respond to the human as helpfully and accurately as possible. You have access to the following tools:

        {tools}

        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

        Valid "action" values: "Final Answer" or {tool_names}

        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{
        "action": $TOOL_NAME,
        "action_input": $INPUT
        }}
        ```

        Follow this format:

        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        ... (repeat Thought/Action/Observation N times)
        Thought: I know what to respond
        Action:
        ```
        {{
        "action": "Final Answer",
        "action_input": "Final response to human"
        }}

        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''

    human = '''{input}

    {agent_scratchpad}

    (reminder to respond in a JSON blob no matter what)'''

    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
        ]
    )

    retriever_tool = create_retriever_tool(
    retriever = retriever,
    name = "webpages_information", 
    description = "Information from corpuses for webpages stored by the user." 
    )

    tools = [retriever_tool, search_tool] 
    tool_names = [t.name for t in tools]
    print("tools: ", tools)
    print("tool_names: ", tool_names)

    agent = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)

    return agent_executor 

def websearch_chain(llm):
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
        .assign(summmary = itemgetter("answer") | llm | StrOutputParser())
        .pick(["search_query", "answer", "summmary"])
    )

    return chain 

def youtube_chain(llm, retriever):

    # Contextualize question
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # contextualize_q_llm = llm.with_config(tags=["contextualize_q_llm"]) # Tags internally (dnot in response) allow to label model instance for tracing during inference useful for streaming. 
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer question
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context from youtube videos to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use a few sentences to keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    # retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

"""
[snippet: From a perfect uppercut in the welterweight opener to a massive right hook that ended the evening, the first event after UFC 300 was an explosive affair that produced plenty of highlights and gave ..., title: Main Card Results | UFC Fight Night: Nicolau vs Perez, link: https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-fight-night-nicolau-vs-perez?language_content_entity=en], [snippet: See The Fight Results, Watch Post-Fight Interviews With The Main Card Winners And More From UFC 298: Volkanovski vs Topuria, Live From Honda Center In Anaheim By E. Spencer Kyte, on X @spencerkyte ..., title: Main Card Results | UFC 298: Volkanovski vs Topuria - UFC.com, link: https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-298-volkanovski-vs-topuria?language_content_entity=en], [snippet: Ultimate Fighting Championship (UFC) was back at the friendly confines of the Apex last night (Sat., April 27, 2024) in Las Vegas, Nevada for UFC Vegas 91. Headlining the event was a Flyweight ..., title: UFC Vegas 91 results: Biggest winners, loser from 'Nicolau vs. Perez ..., link: https://www.mmamania.com/2024/4/28/24143415/ufc-vegas-91-results-biggest-winners-loser-nicolau-perez-last-night-espn-mma], [snippet: Esteban Ribovics (29-28, 29-28, 29-28) defeats Kamuela Kirk by unanimous decision . Esteban Ribovics and Kamuela Kirk set the Fight of the Night bar high right out of the gate on Saturday night ..., title: UFC 290: Volkanovski vs Rodriguez Final Results - UFC.com, link: https://www.ufc.com/news/ufc-290-volkanovski-vs-rodriguez-results-highlights-winner-interviews?language_content_entity=en]
"""

"""
> Entering new AgentExecutor chain...
Question: How to run agent as iterator?

Thought: The agent needs to be run as an iterator to process a sequence of inputs.

Action:
```
{
  "action": "webpages",
  "action_input": "iterator"
}
```
Running Agent as an Iterator | 🦜️🔗 LangChain

Add chat history | 🦜️🔗 LangChain

us out by providing feedback on this documentation page:PreviousStreamingNextReturning Structured OutputCommunityDiscordTwitterGitHubPythonJS/TSMoreHomepageBlogYouTubeCopyright © 2024 LangChain, Inc.

class_=("post-content", "post-title", "post-header")        )    ),)docs = loader.load()text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)splits = text_splitter.split_documents(docs)vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())retriever = vectorstore.as_retriever()### Contextualize question ###contextualize_q_system_prompt = """"Given a chat history and the latest user question \which might reference context in the chat history, formulate a standalone question \which can be understood without the chat history. Do NOT answer the question, \just reformulate it if needed and otherwise return it as is.""""contextualize_q_prompt = ChatPromptTemplate.from_messages(    [        ("system", contextualize_q_system_prompt),        MessagesPlaceholder("chat_history"),        ("human", "{input}"),    ])history_aware_retriever = create_history_aware_retriever(    llm, retriever, contextualize_q_prompt)### Answer questionHere is the response:

Action:
```
{
  "action": "Final Answer",
  "action_input": "To run the agent as an iterator, you need to use the `create_history_aware_retriever` function, which takes the language model, the retriever, and a prompt template as inputs. The prompt template should be used to contextualize the question, and the retriever should be used to retrieve relevant documents. The function will return a retriever that is aware of the chat history."
}
```

> Finished chain.
"{'input': 'how to run agent as iterator?', 'output': 'To run the agent as an iterator, you need to use the `create_history_aware_retriever` function, which takes the language model, the retriever, and a prompt template as inputs. The prompt template should be used to contextualize the question, and the retriever should be used to retrieve relevant documents. The function will return a retriever that is aware of the chat history.'}"

"""

"""
URLs:
- https://www.datacamp.com/tutorial/how-transformers-work
- https://medium.com/@puneetthegde22/mamba-architecture-a-leap-forward-in-sequence-modeling-370dfcbfe44a

Prompt: explain difference between transformers and mamba
"""
