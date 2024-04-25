from operator import itemgetter

from langchain import hub
from langchain.agents import AgentExecutor, AgentType, Tool, tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent, create_csv_agent
from langchain.tools import BaseTool
from langchain.tools.render import render_text_description_and_args
from langchain_experimental.tools import PythonAstREPLTool

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable

class DataFrameToolChain:
    def __init__(self, dataframes, llm):
        self.dataframes = dataframes # Dictionary of pandas dataframes
        self.llm = llm
        self.tools = [PythonAstREPLTool(locals=self.dataframes)] # List of tools
        self.llm_with_tools = self.llm.bind_tools(self.tools) # Integrate tools with LLM that support tool calling

    def call_tools(self, msg: AIMessage) -> dict:
        """Simple sequential tool calling helper."""

        tool_map = {tool.name: tool for tool in self.tools} # Dictionary of all tools
        tool_calls = msg.tool_calls.copy() # Copy list of tool calls
        print(tool_calls)

        for tool_call in tool_calls:
            tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"]) # Execute tool with provided parameters and add to output key in dictionary
        
        return tool_calls

    def get_dfchain(self):
        df_template = """```python
        {df_name}.head().to_markdown()
        >>> {df_head}
        ```"""

        # Format dataframes for prompt
        df_context = "\n\n".join(
            df_template.format(df_head=df.head().to_markdown(), df_name=df_name)
            for df_name, df in self.dataframes.items()
        )
        print(df_context)

        system = f"""You have access to a number of pandas dataframes. \
        Here is a sample of rows from each dataframe and the python code that was used to generate the sample:

        {df_context}

        Given a user question about the dataframes, write the Python code to answer it. \
        Don't assume you have access to any libraries other than built-in Python ones and pandas. \
        Make sure to refer only to the variables mentioned above."""
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system), 
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ]
        )

        chain = prompt | self.llm_with_tools | self.call_tools

        return chain

# chain = DataFrameToolChain(dataframes, llm)
# result = chain.invoke({"question":xyz, "chat_history":memory.load_memory_variables({})})

def df_agent(llm, paths, question):
    agent = create_csv_agent(
        llm,
        paths, 
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent.invoke(question)
