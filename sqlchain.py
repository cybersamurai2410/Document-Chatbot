from operator import itemgetter

from langchain_community.utilities import SQLDatabase 
from langchain_community.agent_toolkits import create_sql_agent
import mysql.connector 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}" # mysql driver
    print(db_uri)

    try:
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        print(f"An error occurred while trying to connect to the database: {e}")

def get_sqlchain(db, llm):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()
    
    query_chain = RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser()

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
    prompt = ChatPromptTemplate.from_template(template)

    # chain = (
    #     RunnablePassthrough.assign(query=query_chain).assign(
    #     schema=lambda _: db.get_table_info(),
    #     response=lambda vars: db.run(vars["query"]),
    #     )
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    chain = (
        RunnablePassthrough.assign(query=query_chain)
        .assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        .assign(
            answer={
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "query": itemgetter("query"),
                "schema": itemgetter("schema"),
                "response": itemgetter("response"),
            } | prompt | llm
        )
        .pick(["query", "response", "answer"])
    )

    return chain 

# agent_executor = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)

'''
how many albums are there?
content='There are 3 albums in the database.' response_metadata={'token_usage': {'completion_time': 0.007283877, 'completion_tokens': 10, 'prompt_time': 0.541375225, 'prompt_tokens': 3022, 'queue_time': None, 'total_time': 0.548659102, 'total_tokens': 3032}, 'model_name': 'llama3-8b-8192', 'system_fingerprint': 'fp_179b0f92c9', 'finish_reason': 'stop', 'logprobs': None} id='run-9ebda93c-cea8-45ea-9e2d-080202972849-0'
'''
