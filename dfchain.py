from operator import itemgetter
import streamlit as st

from langchain import hub
from langchain.agents import AgentExecutor, AgentType, Tool, tool, create_structured_chat_agent, create_react_agent
from langchain.tools import BaseTool
from langchain.tools.render import render_text_description_and_args
from langchain_experimental.tools import PythonAstREPLTool

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

@tool
def get_dfchain(dataframes, llm):
    """Execute python code using pandas datframe."""

    pytool = PythonAstREPLTool(locals=dataframes)
    # llm_with_tools = llm.bind_tools([pytool], tool_choice=tool.name) # Supported by models with function calling
    # chain = prompt | llm_with_tools | JsonOutputParser | tool
    
    df_template = """```python
    {df_name}.head().to_markdown()
    >>> {df_head}
    ```"""

    df_context = "\n\n".join(
        df_template.format(df_head=df.head().to_markdown(), df_name=df_name)
        for df_name, df in dataframes.items()
    )
    print(df_context)

    system = f"""You have access to a number of pandas dataframes. \
    Here is a sample of rows from each dataframe and the python code that was used to generate the sample:

    {df_context}

    Given a user question about the dataframes, write the Python code to answer it. \
    Don't assume you have access to any libraries other than built-in Python ones and pandas. \
    Make sure to refer only to the variables mentioned above."""
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

    explainsolution_prompt = PromptTemplate.from_template(
        """Tell what the answer is first then explain briefly what calculation was used and provide analysis of the answer. \
            Do not mention any coding syntax terms or variable names from the code!
        
        Code:
        {code}
        
        Answer:
        {answer}"""
    )

    code_chain = prompt | llm | StrOutputParser()
    solution_chain = (
    RunnablePassthrough.assign(code=code_chain)
    .assign(answer=itemgetter("code") | pytool)
    .assign(explanation = {"code": itemgetter("code"), "answer": itemgetter("answer")} | explainsolution_prompt | llm) 
    .pick(["code", "answer", "explanation"])
    )
    # solution = solution_chain.invoke({"question": question})

    return solution_chain

@tool
def df_charts(dataframes):
    """Display visualisations from pandas dataframe."""
    pass

print("Executing chain...")
csv_tools = [get_dfchain]
rendered_tools = render_text_description_and_args(csv_tools)
prompt = hub.pull("hwchase17/react-json") # react-multi-input-json

def execute_csvagent(llm, question):
    agent = create_react_agent(prompt, llm, csv_tools) 
    agent_executor = AgentExecutor(agent=agent, tools=csv_tools, verbose=True, max_iterations=5)
    result = agent_executor.invoke({"input": question})

    return result
