import openai
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from config import CHUNK_SIZE, BASE_URL, OPENAI_API_KEY
from embedding_utils import get_embedding

# библиотека OpenAI по умолчанию ищет ключ в переменных окружения, поэтому загружаем их
load_dotenv()

client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY
)


def split_file_into_chunks(file_path: Path, chunk_size: int = CHUNK_SIZE) -> List[Tuple[str, str]]:

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk_text = "".join(lines[i:i+chunk_size])
        chunk_id = f"{file_path}:{i+1}-{min(i+chunk_size, len(lines))}"
        chunks.append((chunk_text, chunk_id))

    return chunks

def embed_chunks(chunks: List[Tuple[str, str]]) -> List[dict]:
    embeddings = []
    for text, chunk_id in tqdm(chunks, desc="Embedding chunks"):
        embedding = get_embedding(text)
        embeddings.append({
            "vector": embedding,
            "text": text,
            "source": chunk_id
        })
    return embeddings
