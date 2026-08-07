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
    model="openai/gpt-oss-120b",
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
                f"Taxi {state['taxi_id']} anomaly detected. "
                f"ANOMALY TYPE: {state['anomaly_type']}. "
                f"ANOMALY SPEED AT DETECTION: {state['speed']:.1f} m/s — use this as speed_detected. "
                f"UNCERTAINTY: {state['uncertainty']:.2f}. "
                f"Severity rule: if speed > 80 m/s OR type is position_jump → severity MUST be HIGH. "
                f"Query vehicle history to understand driving pattern. "
                f"Query maintenance to check if any service is overdue. "
                f"Search similar past anomalies for context. "
                f"Based on ALL gathered data, respond ONLY with this JSON, no text outside it:\n"
                f'{{"taxi_id": "{state["taxi_id"]}", '
                f'"anomaly_type": "{state["anomaly_type"]}", '
                f'"severity": "HIGH or MEDIUM or LOW", '
                f'"speed_detected": {state["speed"]:.1f}, '
                f'"diagnosis": "2-3 sentences: what happened, likely cause, pattern from history", '
                f'"recommendation": "specific action with urgency: immediate/within 24h/within week", '
                f'"maintenance_status": "up to date/overdue — mention next due service if relevant", '
                f'"confidence": "HIGH if similar past anomalies found, MEDIUM if first occurrence, LOW if uncertain"}}'
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