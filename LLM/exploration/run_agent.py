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
    print("     School Zoning Exploration Agent")
    print("=" * 60)
    print()
    print("I'll help you explore different school zoning proposals.")
    print("Tell me what matters most to you, and I'll help find solutions")
    print("that match your priorities.")
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

    clusters_result = agent.get_initial_clusters()
    if clusters_result:
        print(f"Agent: {clusters_result['text']}\n")
        while True:
            try:
                choice = input("You (enter group number): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                sys.exit(0)
            if choice.isdigit():
                result = agent.select_cluster(int(choice))
                print(f"\nAgent: {result['text']}\n")
                break
            print("Please enter a group number.")
    else:
        result = agent.chat("Show me the current balanced solution.")
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
