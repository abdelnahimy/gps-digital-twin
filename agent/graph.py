import sys
import os
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from agent.tools import query_vehicle_history, query_maintenance, find_similar_anomalies, get_fleet_summary
from agent.prompts import SYSTEM_PROMPT

tools = [query_vehicle_history, query_maintenance, find_similar_anomalies, get_fleet_summary]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT
)

def run_agent(question: str) -> str:
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content