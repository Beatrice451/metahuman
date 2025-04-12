from openai import OpenAI
from config import BASE_URL, OPENAI_API_KEY

client = OpenAI(
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY
)

def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Строит промпт для генерации ответа, включая релевантные чанки.
    """
    context_text = "\n\n".join(
        f"[{chunk['source']}]\n{chunk['text']}" for chunk in context_chunks
    )
    return f"""Ты — ассистент, помогающий понять содержимое репозитория. Используй приведённый контекст для ответа.

Контекст:
{context_text}

Вопрос: {question}
Ответ:"""

def generate_answer(prompt: str) -> str:
    """
    Отправляет промпт в LLM и возвращает сгенерированный ответ.
    """
    response = client.chat.completions.create(
        model="lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF",  # заменяется на название локальной модели, если нужно
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
