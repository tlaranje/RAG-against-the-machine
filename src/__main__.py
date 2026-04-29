from tqdm import tqdm
import fire


# 1. Exemplo tqdm (Barra de progresso)
def demonstrar_tqdm():
    print("\n--- Demonstração tqdm ---")
    import time
    for i in tqdm(range(10), desc="Processando tarefas"):
        time.sleep(0.1)


# 2. Exemplo Transformers (Análise de Sentimento)
def demonstrar_transformers():
    print("\n--- Demonstração Transformers ---")
    from transformers import pipeline
    # Carrega um pipeline simples de análise de sentimento
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    resultado = classifier("I love learning about AI libraries!")[0]
    print("Texto: 'I love learning about AI libraries!'")
    print(f"Resultado: {resultado['label']} (Score: {resultado['score']:.4f})")


# 3. Exemplo ChromaDB (Banco de Dados Vetorial)
def demonstrar_chromadb():
    print("\n--- Demonstração ChromaDB ---")
    import chromadb
    client = chromadb.Client()
    collection = client.create_collection(name="minha_colecao")

    collection.add(
        documents=[
            "Este é um documento sobre Python",
            "Este é um documento sobre Java"
        ],
        metadatas=[{"source": "py_file"}, {"source": "java_file"}],
        ids=["id1", "id2"]
    )

    results = collection.query(
        query_texts=["Qual fala de Python?"], n_results=1
    )
    print(f"Documento mais próximo: {results['documents'][0][0]}")


# 4. Exemplo LangChain (Prompt Template)
def demonstrar_langchain():
    print("\n--- Demonstração LangChain ---")
    from langchain_core.prompts import PromptTemplate
    template = "Diz-me um facto engraçado sobre {topico}."
    prompt = PromptTemplate(input_variables=["topico"], template=template)
    print(f"Prompt Gerado: {prompt.format(topico='Gatos')}")


# 5. Exemplo BM25s (Busca por relevância)
def demonstrar_bm25s():
    print("\n--- Demonstração BM25s ---")
    import bm25s
    corpus = [
        "O rato roeu a rolha", "O gato caçou o rato", "O cão ladrou ao gato"
    ]
    # Criar o modelo e indexar
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))

    query = "rato"
    results, scores = retriever.retrieve(bm25s.tokenize(query), k=2)
    print(f"Busca por '{query}': {results[0]}")


# 6. Exemplo DSPy (Assinatura de Programa)
def demonstrar_dspy():
    print("\n--- Demonstração DSPy ---")
    import dspy
    # Definir uma assinatura simples (Input -> Output)

    class ResumoSimples(dspy.Signature):
        """Resume um texto longo de forma concisa."""
        texto = dspy.InputField()
        resumo = dspy.OutputField()

    print(
        "DSPy configurado. Exemplo de "
        "Assinatura definida para 'ResumoSimples'."
    )


# 7. Exemplo Fire (Transformar funções em CLI)
# O Fire é usado no final para permitir correr estas funções via terminal
def main(lib="todas"):
    """
    Corre exemplos das bibliotecas.
    Uso: python script.py --lib=transformers
    """
    if lib == "tqdm" or lib == "todas":
        demonstrar_tqdm()
    if lib == "transformers" or lib == "todas":
        demonstrar_transformers()
    if lib == "chromadb" or lib == "todas":
        demonstrar_chromadb()
    if lib == "langchain" or lib == "todas":
        demonstrar_langchain()
    if lib == "bm25s" or lib == "todas":
        demonstrar_bm25s()
    if lib == "dspy" or lib == "todas":
        demonstrar_dspy()
    print()


if __name__ == "__main__":
    # O Fire expõe a função main para a linha de comandos
    fire.Fire(main)
