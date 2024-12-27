from agents import GuardAgent
from agents import AgentProtocol
import os

def main():
    pass

if __name__ == "__main__":
    guard_agent = GuardAgent()

    # print(isinstance(guard_agent, AgentProtocol))

    messages = []
    
    while True:
        os.system("cls" if os.name == "nt" else "clear")

        print("\n\n Print messages ............")

        for message in messages:
            print(f"{message["role"]} : {message["content"]}")

        # Get user input
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get guard agent's response
        guard_agent_response = guard_agent.get_response(messages)
        print("GUARD AGENT OUTPUT: ", guard_agent_response)
        messages.append(guard_agent_response)