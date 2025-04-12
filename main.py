from repo_util import clone_repo, get_all_code_files
from document_indexer import split_file_into_chunks, embed_chunks
from retriever import retrieve_relevant_chunks

def main():
    print("🔗 Git RAG System")
    # repo_url = input("Введите ссылку на публичный репозиторий: ").strip()
    repo_url = "https://github.com/Beatrice451/metahuman"

    # Шаг 1. Клонируем репозиторий
    print("📥 Клонируем репозиторий...")
    repo_path = clone_repo(repo_url)

    # Шаг 2. Получаем список исходных файлов
    code_files = get_all_code_files(repo_path)
    print(f"📂 Найдено исходных файлов: {len(code_files)}")

    # Шаг 3. Разбиваем файлы на чанки
    chunks = []
    for file in code_files:
        chunks.extend(split_file_into_chunks(file))

    print(f"✂️ Получено чанков: {len(chunks)}")

    # Шаг 4. Создаём embedding для всех чанков
    embeddings = embed_chunks(chunks)
    print("✅ Embedding завершены")

    # Шаг 5. Вопрос пользователя
    while True:
        question = input("\n💬 Введите вопрос (или 'exit' для выхода): ").strip()
        if question.lower() in ["exit", "quit"]:
            break

        # Шаг 6. Получение релевантных чанков
        top_chunks = retrieve_relevant_chunks(question, embeddings)

        print("\n🔎 Наиболее релевантные фрагменты:")
        for chunk in top_chunks:
            print(f"\n📄 {chunk['source']}")
            print(chunk["text"])

if __name__ == "__main__":
    main()
