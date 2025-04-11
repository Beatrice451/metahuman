import openai
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# Размер чанка (в токенах можно позже), пока просто по строкам
CHUNK_SIZE = 30

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
    """Получает embedding для каждого чанка"""
    embeddings = []
    for text, chunk_id in tqdm(chunks, desc="Embedding chunks"):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embeddings.append({
                "vector": response.data[0].embedding,
                "text": text,
                "source": chunk_id
            })
        except Exception as e:
            print(f"Ошибка при обработке {chunk_id}: {e}")
    return embeddings
