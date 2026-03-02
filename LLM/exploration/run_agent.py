"""
CLI runner for the School Zoning Exploration Agent.

Run with:
    cd /home/kumarc/sfusd/SFUSD-Zone
    python -m LLM.exploration.run_agent
"""

import sys
from pathlib import Path

from .zoning_agent import ZoningAgent


DEFAULT_CSV_PATH = "~/sfusd-local-data/zones/SFUSD/local_runs/new_benchmarks_test/summary.csv"


def main():
    """Run the interactive zoning exploration agent."""
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("     Understanding Elementary School Zones")
    print("=" * 60)
    print()
    print("This tool helps you explore draft zone maps. Your feedback will")
    print("inform rules and guidelines for the next set of zone maps.")
    print()
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'reset' to clear all filters and start over.")
    print()
    print("-" * 60)
    
    try:
        agent = ZoningAgent(csv_path)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    print()
    
    # Opening: three messages — (1) how the tool works, (2) solution + clusters, (3) choose cluster prompt
    first_response = agent.chat("Begin.")
    print(f"\nAgent: {first_response}\n")
    second_response = agent.chat("Now show me the current solution and the clusters.")
    print(f"\nAgent: {second_response}\n")
    print("\nAgent: Which cluster would you like to explore? Just tell me the cluster number (1, 2, or 3).\n")
    result = agent.chat(
        "Please show me the current balanced solution and explain what each metric means briefly."
    )
    print(f"\nAgent: {result['text']}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break
        
        if user_input.lower() == "reset":
            agent.reset_filters()
            print("Filters reset. Let's start fresh.\n")
            continue
        
        try:
            result = agent.chat(user_input)
            print(f"\nAgent: {result['text']}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
