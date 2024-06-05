from langchain_community.utilities import SQLDatabase 
from langchain_community.agent_toolkits import create_sql_agent

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"

    return SQLDatabase.from_uri(db_uri)

# agent_executor = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)
