from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

model = ChatOllama(model="llama3")

data = pd.read_csv("employees.csv")  # type: ignore

agent = create_pandas_dataframe_agent(
    llm=model,
    df=data,
    verbose=True,
    include_df_in_prompt=True,
    allow_dangerous_code=True,
)

message = "Find employee with id 4 using the provided dataframe"

result: BaseMessage = model.invoke(message)

result.pretty_print()
