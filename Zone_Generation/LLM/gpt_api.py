import requests

# Your OpenAI API key
api_key = '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'

payload = {
    'model': 'gpt-3.5-turbo-1106',  # Specify the model here
    'prompt': 'Translate the following text to French: Hello, how are you?',
    'temperature': 0.5,
    'max_tokens': 100
}

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

response = requests.post('https://api.openai.com/v1/completions',
                         json=payload, headers=headers)

# Print the response text (the translated text)
print(response.text)