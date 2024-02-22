from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import  StructuredTool
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder



llm = ChatOpenAI(openai_api_key='123', organization="gluon-meson",
openai_api_base='https://0972-152-101-166-135.ngrok-free.app', model='gpt-3.5-turbo', temperature=0, streaming=False)

embeddings = OpenAIEmbeddings(openai_api_key='123', organization="gluon-meson",
openai_api_base='https://0972-152-101-166-135.ngrok-free.app', model='text-embedding-ada-002')

class PowerCalculation(BaseModel):
    """Call this to calculate the power of a number to the nth power"""
    base: float = Field(description="The base number")
    exponent: int = Field(description="The exponent (power)")

def calculate_power(base: float, exponent: int) -> float:
    """Calculate the power of a number to the nth power"""
    result = base ** exponent
    return result


class SquareCalculation(BaseModel):
    """Call this to calculate the square of a number"""
    number: float = Field(description="The number to be squared")

def calculate_square(number: float) -> float:
    """Calculate the square of a number"""
    result = number ** 2
    return result

class Multiplication(BaseModel):
    """Call this to perform multiplication of two numbers"""
    num1: float = Field(description="The first number")
    num2: float = Field(description="The second number")

def calculate_multiplication(num1: float, num2: float) -> float:
    """Perform multiplication of two numbers"""
    result = num1 * num2
    return result


agent_tools=[
    StructuredTool.from_function(func=calculate_power, name="PowerCalculation", description="Call this to calculate the power of a number to the nth power"),
    StructuredTool.from_function(func=calculate_square, name="SquareCalculation", description="Call this to calculate the square of a number"),
    StructuredTool.from_function(func=calculate_multiplication, name="Multiplication", description="all this to perform multiplication of two numbers")
]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are math expert, answer me using a friendly tone"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


agent = create_openai_tools_agent(llm, agent_tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True)


result = agent_executor.invoke({"input":"Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"})
print(result["output"])





