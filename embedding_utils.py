import openai
from config import BASE_URL, OPENAI_API_KEY

client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY
)


def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-nomic-embed-text-v1.5-embedding",
        input=text
    )
    return response.data[0].embedding
