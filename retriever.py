import openai
import numpy as np
from typing import List, Dict

def get_question_embedding(question: str) -> List[float]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    return response.data[0].embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def retrieve_relevant_chunks(question: str, indexed_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    q_embedding = get_question_embedding(question)
    scored_chunks = []

    for chunk in indexed_chunks:
        sim = cosine_similarity(q_embedding, chunk["vector"])
        scored_chunks.append((sim, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
    return top_chunks
