from operator import itemgetter
import os
from pydantic import BaseModel, Field
from typing import List 

from langchain import hub
from langchain.agents import AgentExecutor, AgentType, Tool, tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent, create_csv_agent
from langchain.tools.render import render_text_description_and_args
from langchain_experimental.tools import PythonAstREPLTool

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable

class DataFrameToolChain:
    def __init__(self, dataframes, llm):
        self.dataframes = dataframes # Dictionary of pandas dataframes
        self.llm = llm
        self.tools = [PythonAstREPLTool(locals=self.dataframes)] # List of tools; include another tool for general queries
        self.llm_with_tools = self.llm.bind_tools(self.tools) # Integrate tools with LLM that support tool calling

    def call_tools(self, msg: AIMessage) -> dict:
        """Simple sequential tool calling helper."""

        print("msg: ", msg)
        tool_map = {tool.name: tool for tool in self.tools} # Dictionary of all tools
        tool_calls = msg.tool_calls.copy() # Copy list of tool calls
        
        for tool_call in tool_calls:
            tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"]) # Execute tool with provided parameters and add to output key in dictionary
            
            if tool_call["output"]:
                prompt = f"""Provide and explanation for the result given the following code:
                {tool_call["args"]["query"]}

                Result:
                {tool_call["output"]}
                
                The answer should explain the reasoning to get the result and not explain the code syntax."""
                tool_call["explanation"] = self.llm.invoke(prompt)
        
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
        # print(df_context)

        system = f"""You have access to a number of pandas dataframes. \
        Here is a sample of rows from each dataframe and the python code that was used to generate the sample:

        {df_context}

        Write the Python code to answer the user prompt that can be requesting data analytics and visualizations about the dataframes. \
        You already have access to the dataframes as local variables: {self.dataframes.keys()}. \
        You can use the library matplotlib for generating graphs if data visualization is relevant. \
        Give the graph a unique file name then save to directory called 'Graphs' e.g. plt.savefig('Graphs/file_name.png') and attach to final answer in markdown format. \
        This is the current list of image files in the directory 'Graphs': {os.listdir('Graphs')}

        Return the final answer from the code as python dictionary about the results.""" 
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system), 
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ]
        ) # .partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.llm_with_tools | self.call_tools

        return chain

# chain = DataFrameToolChain(dataframes, llm)
# result = chain.invoke({"question":xyz, "chat_history":memory.load_memory_variables({})["history"]})

def df_agent(llm, paths, question):
    agent = create_csv_agent(
        llm,
        paths, 
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent.invoke(question)
