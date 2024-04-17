from operator import itemgetter

from langchain import hub
from langchain.agents import AgentExecutor, AgentType, Tool, tool, load_tools, create_structured_chat_agent, create_react_agent
from langchain.tools.render import render_text_description
from langchain_experimental.tools import PythonAstREPLTool

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

@tool
def get_dfchain(dataframes):
    """Execute python code using pandas datframe."""

    pytool = PythonAstREPLTool(locals=dataframes)
    # tools = [pytool]

    df_template = """```python
    {df_name}.head().to_markdown()
    >>> {df_head}
    ```"""

    df_context = "\n\n".join(
        df_template.format(df_head=df.head().to_markdown(), df_name=df_name)
        for df_name, df in dataframes.items()
    )
    print(df_context)

    chain = ''
    solution = chain.invoke()

    return solution

@tool
def df_stats(datframes):
    """Get statistics of pandas dataframe."""
    pass

csv_tools = [get_dfchain]
rendered_tools = render_text_description(csv_tools)
prompt = hub.pull("hwchase17/react")

def execute_csvagent(llm, question):
    agent = create_react_agent(prompt, llm, csv_tools) # ReAct - Synergizing Reasoning and Acting in Language Models
    agent_executor = AgentExecutor(agent=agent, tools=csv_tools, verbose=True, max_iterations=5)
    result = agent_executor.invoke({"input": question})

    return result
