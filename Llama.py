from llama_cpp import Llama
import os

os.environ['LLAMA_LOG_LEVEL'] = 'error'
os.environ['LLAMA_DEBUG'] = '0'
os.environ['LLAMA_VERBOSE'] = '0'
os.environ['LLAMA_SUPPRESS_LOGS'] = '1'

llm = Llama.from_pretrained(
    repo_id="althayr/Gemma-3-Gaia-PT-BR-4b-it-Q8_0-GGUF",
    filename="Gemma-3-Gaia-PT-BR-4b-it_Q8_0.gguf",
    verbose=False,
    n_ctx=8192  
)

messages = []

while True:
    user = input("\n")
    
    messages.append({
        "role": "user",
        "content": user
    })
    
    stream = llm.create_chat_completion(
        messages=messages, 
        stream=True
    )

    bot_response = ""
    for chunk in stream:
        if chunk['choices'][0]['delta'].get('content'):
            content = chunk['choices'][0]['delta']['content']
            print(content, end='', flush=True)
            bot_response += content
    
    messages.append({
        "role": "assistant",
        "content": bot_response
    })
