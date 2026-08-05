import sys, time, json, re
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from dotenv import load_dotenv
load_dotenv()
import os

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from agent.tools import query_vehicle_history, query_maintenance, find_similar_anomalies, get_fleet_summary
from agent.prompts import SYSTEM_PROMPT

tools = [query_vehicle_history, query_maintenance, find_similar_anomalies, get_fleet_summary]

TEST_QUESTION = (
    "Taxi 20000520 anomaly: type=position_jump, speed=45.0 m/s. "
    "Query history and maintenance. Respond ONLY with JSON."
)

def test_llm(name, llm):
    agent = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
    
    start = time.time()
    result = agent.invoke({"messages": [("user", TEST_QUESTION)]})
    elapsed = time.time() - start
    answer = result["messages"][-1].content
    
    # check JSON compliance
    try:
        match = re.search(r'\{.*\}', answer, re.DOTALL)
        parsed = json.loads(match.group()) if match else None
        json_ok = parsed is not None
    except:
        json_ok = False
    
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Time:  {elapsed:.2f}s")
    print(f"JSON:  {'✅' if json_ok else '❌'}")
    print(f"Response:\n{answer[:300]}")
    return {"model": name, "time": elapsed, "json_ok": json_ok, "answer": answer}

# Test Groq
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Test Gemini
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

results = []
results.append(test_llm("Groq llama-3.3-70b", groq_llm))
results.append(test_llm("Gemini 2.0 Flash", gemini_llm))

print(f"\n{'='*50}")
print("SUMMARY")
for r in results:
    print(f"{r['model']:25} | Time: {r['time']:.2f}s | JSON: {'✅' if r['json_ok'] else '❌'}")