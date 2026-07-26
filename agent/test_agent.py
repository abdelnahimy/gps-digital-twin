import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from agent.graph import run_agent

answer = run_agent("What is the maintenance status of taxi 20000154?")
print(answer)