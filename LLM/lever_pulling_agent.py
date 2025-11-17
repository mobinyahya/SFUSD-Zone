import json

from ollama import Client
from openai import OpenAI
from pydantic import BaseModel


# class OptimizationConfig(BaseModel):
#     frl_bound: float
#     diversity_bound: float
#     population_bound: float
#     zone_school_bound: int


class LeverPullingAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url = 'http://100.108.82.68:11434/v1/',
            api_key='ollama', # required, but unused
        )

    def query(self, text: str):
        response = self.client.chat.completions.parse(
            model='gemma3:27b',
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized legal assistant. Provide only factual information and cite sources where possible."
                },
                {
                    'role': 'user',
                    'content': text,
                },
            ],
            response_format=OptimizationConfig,
        )
        return response.choices[0].message.parsed
        # return json.loads(completion.choices[0].message.parsed)

if __name__ == "__main__":
    agent = LeverPullingAgent()
    response = agent.query("What is the capital of France?")
    print(response)
    print(type(response))
