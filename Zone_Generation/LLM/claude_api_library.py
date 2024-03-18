import anthropic
from config import ANTHROPIC_API_KEY


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

Prompt = "how far is the sun. Return an single integer number"

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": Prompt}
    ]
)
print(message.content[0].text)
