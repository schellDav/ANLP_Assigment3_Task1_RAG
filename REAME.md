# Homework 3: Simple RAG System from Scratch

This repository contains a lightweight Retrieval-Augmented Generation (RAG) system implemented in Python. The system reads text datasets, creates vector embeddings for each text chunk (line) using a local embedding model, performs in-memory cosine similarity search to retrieve relevant context, and uses a local language model (LLM) to generate grounded, hallucination-free answers via Ollama.

## Prerequisites & Setup

Ensure you have Python 3.8+ installed on your system.

### 1. Install Dependencies
Install the required packages using pip:
```bash
pip install ollama
```
Also pull the following modules:
```
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

### 2. Running the Program:

To run the RAG system, execute the `demo.py` script:
```
python demo.py
```
After that:

1. **Load Knowledge Base:** You will be prompted to specify the database file to load. Press Enter to default to `cat-facts.txt`, or enter `cat-db-fiction-confusing.txt`.  

2. **Set Top-N Chunks:** Enter the number of chunks you wish to retrieve (default is 3).

3. **Ask Questions:** Enter your query. The pipeline will output:  
    - The retrieved knowledge chunks with their calculated cosine similarity scores.  
    - A grounded streamed response from the LLM based strictly on that context.

4. **Exit:** Type exit or quit to close the interactive session.

## Evaluation

Below is the evaluation of the RAG system under the two database configurations utilizing $top\_n = 3$

| File used | Question no. | `top_n` | Generated answer | Correct? | If incorrect, how could it be fixed? |
| :--- | ---: | ---: | :--- | :--- | :--- |
| `cat-facts.txt` | 1 | 3 | The additional organ that allows cats to smell is the Jacobson's organ, which is located in the upper surface of the mouth. | Yes | --- |
| `cat-facts.txt` | 2 | 3 | Since a cat spends 2/3 of every day sleeping and they are only awake for 1/3 of their life, we can calculate the number of waking years as follows:<br><br>1 (the total time awake) - 0.5 (the proportion of time asleep) = 0.5<br><br>Multiply by 9 (the cat's age in human years):<br><br>0.5 × 9 = approximately 4.5 human years.<br><br>So, a nine-year-old cat has been awake for about 4.5 years. | No | The model hallucinates math despite the exact answer ("three years") being in the retrieved context. This can be fixed by improving the prompt to explicitly instruct the model to extract exact figures from the text rather than calculating them itself. |
| `cat-db-fiction-confusing.txt` | 3 | 3 | The Riverside Luna must eat rabbit-based food because her record lists a chicken allergy. | No | The answer is incomplete because it missed the collar color (which was successfully retrieved). This can be fixed by improving the prompt to explicitly instruct the model to address all parts of a multi-part question. |
| `cat-db-fiction-confusing.txt` | 4 | 3 | Based on the given context, we can infer that:<br><br>• Jupiter has no activity restriction (meaning she is free to attend events)<br>• A veterinary note dated 2026-05-01 says Juniper must rest for seven days and cannot attend adoption events during that period.<br><br>There is insufficient evidence in the knowledge base to determine whether Juniper's sister Jupiter may or may not attend the adoption event. Therefore, we cannot conclude with certainty that Juniper will be attending. | No | The model confused the entities "Juniper" and "Jupiter" despite retrieving the correct veterinary note. This can be fixed by improving the prompt to strictly enforce exact entity matching when evaluating evidence. |
