from ollama import AsyncClient

async def generate_from_ollama(model: str, prompt: str) -> str:
    """
    Asynchronously calls the local Ollama server on the custom port.
    """
    # Wichtig: Nutzen Sie hier Ihren Custom-Port 11435!
    client = AsyncClient(host="http://127.0.0.1:11435")
    
    # Generate the response asynchronously
    response = await client.generate(model=model, prompt=prompt)
    return response['response']