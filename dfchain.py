from operator import itemgetter

from langchain.agents import AgentExecutor, AgentType
from langchain.tools.render import render_text_description

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

def get_dfchain(dataframes):
    pytool = PythonAstREPLTool(locals=dataframes)
    tools = [pytool]

    df_template = """```python
    {df_name}.head().to_markdown()
    >>> {df_head}
    ```"""

    df_context = "\n\n".join(
        df_template.format(df_head=df.head().to_markdown(), df_name=df_name)
        for df_name, df in dataframes.items()
    )
    print(df_context)
