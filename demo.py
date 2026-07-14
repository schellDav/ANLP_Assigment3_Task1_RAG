import math
import sys
import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

VECTOR_DB = []

def cosine_similarity(a, b):
    """Return the cosine similarity between vectors a and b."""
    if not a or not b or len(a) != len(b):
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

# Simple tests for cosine_similarity
assert math.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)
assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)

def add_chunk_to_database(chunk):
    """Request an embedding and append (chunk, embedding) to VECTOR_DB."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
        # Extract the first embedding from the response list
        embedding = response["embeddings"][0]
        VECTOR_DB.append((chunk, embedding))
    except Exception as e:
        print(f"Error embedding chunk: {e}")
        sys.exit(1)

def load_knowledge_base(file_path):
    """Loads a text file, chunks it by line, and populates the VECTOR_DB."""
    global VECTOR_DB
    VECTOR_DB = [] # Reset vector DB when switching files
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # Remove surrounding whitespace and skip empty lines
            dataset = [line.strip() for line in file if line.strip()]
        
        print(f"Loading and embedding {len(dataset)} chunks from {file_path}...")
        for chunk in dataset:
            add_chunk_to_database(chunk)
        print("Vector store built successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'.")
        sys.exit(1)

def retrieve(query, top_n=3):
    """Embed the query and retrieve the top_n most similar chunks."""
    try:
        query_embedding = ollama.embed(
            model=EMBEDDING_MODEL,
            input=query,
        )["embeddings"][0]
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []

    similarities = []
    for chunk, embedding in VECTOR_DB:
        score = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, score))

    # Sort results by similarity in descending order
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_n]

def ask_question(input_query, top_n=3):
    """Runs the retrieval, formats the prompt, and streams the generation."""
    retrieved_knowledge = retrieve(input_query, top_n=top_n)

    print("\nRetrieved knowledge:")
    for chunk, similarity in retrieved_knowledge:
        print(f"- ({similarity:.3f}) {chunk}")

    # Build context string
    context = "\n".join(
        f"- {chunk}" for chunk, _similarity in retrieved_knowledge
    )

    # Construct Grounded Prompt
    instruction_prompt = f"""You are a grounded question-answering assistant.
Use only the context below to answer the user's question.
If the context does not contain enough evidence, say that the answer is not in the knowledge base.
When records conflict, prefer a clearly dated newer record and explain the update briefly.

Context:
{context}
"""

    # Generate and stream the answer
    print("\nAnswer:")
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": input_query},
        ],
        stream=True,
    )

    for response_chunk in stream:
        print(response_chunk["message"]["content"], end="", flush=True)
    print("\n")

def main():
    print("Welcome to the Simple RAG System")
    
    kb_choice = input("Enter the knowledge base file to load (default: cat-facts.txt): ").strip()
    if not kb_choice:
        kb_choice = "cat-facts.txt"
        
    load_knowledge_base(kb_choice)
    
    top_n_str = input("Enter top_n for retrieval (default: 3): ").strip()
    top_n = int(top_n_str) if top_n_str.isdigit() else 3
    
    while True:
        try:
            input_query = input("\nAsk a question (or type 'exit' to quit): ").strip()
            if input_query.lower() in ["exit", "quit"]:
                break
            if not input_query:
                continue
            
            ask_question(input_query, top_n=top_n)
        except KeyboardInterrupt:
            break
            
if __name__ == "__main__":
    main()