import os
from typing import List, Dict, Any

import pandas as pd
from fastmcp import FastMCP
from langchain_openai import ChatOpenAI

from dfchain import DataFrameToolChain
from sqlchain import init_database, get_sqlchain

# === Setup === 

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Load DataFrames
def load_dataframes() -> Dict[str, pd.DataFrame]:
    dfs = {}
    for f in os.listdir("dataframes"):
        if f.endswith(".csv"):
            name = f.replace(".csv", "")
            dfs[name] = pd.read_csv(os.path.join("dataframes", f))
    return dfs

dataframes = load_dataframes()
df_chain = DataFrameToolChain(dataframes, llm).get_dfchain()

# Load SQL connection
db = init_database(
    user="root",             
    password="******", 
    host="localhost",
    port="3306",
    database="my_database" 
)

sql_chain = get_sqlchain(db, llm)

# === MCP Server ===

mcp = FastMCP("DataFrame + SQL AI Server")

@mcp.tool()
def query_dataframes(question: str, chat_history: List[Dict[str, Any]] = None) -> Any:
    """Query local pandas DataFrames with code and explanation."""
    return df_chain.invoke({
        "question": question,
        "chat_history": chat_history or []
    })

@mcp.tool()
def query_sql(question: str, chat_history: List[Dict[str, Any]] = None) -> Any:
    """Query MySQL database with natural language."""
    return sql_chain.invoke({
        "question": question,
        "chat_history": chat_history or []
    })


if __name__ == "__main__":
    mcp.run()
