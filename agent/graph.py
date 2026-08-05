import sys
import os
import threading
import time
import json
import re

sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from dotenv import load_dotenv
load_dotenv()

from storage.mongo_client import diagnoses
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from agent.tools import query_vehicle_history, query_maintenance, find_similar_anomalies, get_fleet_summary
from agent.prompts import SYSTEM_PROMPT

tools = [query_vehicle_history, query_maintenance, find_similar_anomalies, get_fleet_summary]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT
)

_diagnosis_lock = threading.Semaphore(1)
_last_diagnosis_time = 0
_DIAGNOSIS_COOLDOWN = 10  # seconds between diagnoses


def run_agent(question: str) -> str:
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


def auto_diagnose(state: dict) -> None:
    def _diagnose():
        global _last_diagnosis_time

        now = time.time()
        if now - _last_diagnosis_time < _DIAGNOSIS_COOLDOWN:
            return

        if not _diagnosis_lock.acquire(blocking=False):
            return

        try:
            _last_diagnosis_time = time.time()
            question = (
                f"Taxi {state['taxi_id']} anomaly: "
                f"type={state['anomaly_type']}, "
                f"speed={state['speed']:.1f} m/s. "
                f"Query history and maintenance. "
                f"Respond ONLY with this JSON, no other text:\n"
                f'{{"taxi_id": "{state["taxi_id"]}", '
                f'"anomaly_type": "{state["anomaly_type"]}", '
                f'"severity": "HIGH/MEDIUM/LOW", '
                f'"diagnosis": "one sentence", '
                f'"recommendation": "one action", '
                f'"maintenance_status": "up to date/overdue"}}'
            )
            answer = run_agent(question)

            try:
                json_match = re.search(r'\{.*\}', answer, re.DOTALL)
                diagnosis_data = json.loads(json_match.group()) if json_match else {"raw": answer}
            except:
                diagnosis_data = {"raw": answer}

            diagnosis = {
                "taxi_id": state["taxi_id"],
                "timestamp": state.get("timestamp"),
                "anomaly_type": state["anomaly_type"],
                "speed": state["speed"],
                "uncertainty": state["uncertainty"],
                "diagnosis": diagnosis_data
            }
            diagnoses.insert_one(diagnosis)
            print(f"[AGENT] ✅ Diagnosis saved for taxi {state['taxi_id']}")
        except Exception as e:
            print(f"[AGENT] ❌ Failed: {e}")
        finally:
            _diagnosis_lock.release()

    thread = threading.Thread(target=_diagnose, daemon=True)
    thread.start()