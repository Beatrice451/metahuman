from repo_util import clone_repo, get_all_code_files
from document_indexer import split_file_into_chunks, embed_chunks
from retriever import retrieve_relevant_chunks
from qa_engine import build_prompt, generate_answer

REPO_URL = "https://github.com/Beatrice451/metahuman"  # замени на нужный
REPO_PATH = "repo/cloned_repo"
QUESTION = "Что делает этот проект?"  # пример запроса

def main():

    path = clone_repo(REPO_URL, REPO_PATH)

    all_files = get_all_code_files(path)
    chunks = []
    for file in all_files:
        chunks.extend(split_file_into_chunks(file))

    indexed_chunks = embed_chunks(chunks)

    relevant_chunks = retrieve_relevant_chunks(QUESTION, indexed_chunks)

    prompt = build_prompt(QUESTION, relevant_chunks)
    print("Думоем")
    answer = generate_answer(prompt)

    print("\n--- Ответ ---\n")
    print(answer)

if __name__ == "__main__":
    main()
