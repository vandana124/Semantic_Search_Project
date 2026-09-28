# Semantic Search

Semantic search improves search accuracy by understanding the meaning of a search query and the documents being searched. Unlike traditional keyword search, semantic search can retrieve relevant documents even when the exact words in the query are not present.

This project is based on the semantic search example from the [Sentence Transformers](https://github.com/huggingface/sentence-transformers) repository and has been customized for a business-related semantic search task.

## Background

The basic idea behind semantic search is to convert documents and search queries into numerical vector representations called **embeddings**.

At search time:

1. The business documents are converted into embeddings.
2. The search query is converted into an embedding.
3. The query embedding is compared with the document embeddings.
4. The documents with the highest similarity scores are retrieved.

This allows the system to identify documents based on their meaning rather than only matching exact keywords.

## Symmetric vs. Asymmetric Semantic Search

There are two common types of semantic search.

### Symmetric Semantic Search

In symmetric semantic search, the query and the documents have approximately the same amount of information.

For example:

```text
Query:
How can I learn Python online?

Document:
What are the best ways to learn Python on the web?
```

Both the query and document are relatively similar in length and purpose.

### Asymmetric Semantic Search

In asymmetric semantic search, the query is usually short while the documents contain more detailed information.

For example:

```text
Query:
How can a company improve customer loyalty?

Document:
Companies can build customer loyalty by providing personalized services,
offering rewards programs, and responding quickly to customer concerns.
```

The business semantic search system in this project is primarily an **asymmetric semantic search task**, because short business questions are searched against business-related sentences.

## Model

This project uses the following Sentence Transformer model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts sentences into dense vector representations that can be compared to determine semantic similarity.

## Implementation

The project contains a collection of business-related documents and five business search queries.

The documents are encoded using:

```python
model.encode(documents, convert_to_tensor=True)
```

The query is also converted into an embedding:

```python
model.encode(query, convert_to_tensor=True)
```

The embeddings are then passed to the Sentence Transformers semantic search function:

```python
util.semantic_search(
    query_embedding,
    document_embeddings,
    top_k=3
)
```

The system returns the three documents with the highest similarity scores.

## Business Documents

The project contains **24 business-related documents** covering topics such as:

* Customer satisfaction
* Customer retention
* Sales
* Marketing
* Employee productivity
* Business growth
* Supply chain management
* Financial management
* Cybersecurity
* Customer engagement
* Leadership
* Inventory management

## Search Queries

The system is tested using five different business-related queries:

```text
1. How can a company keep its customers loyal?

2. How can businesses improve employee performance?

3. What can companies do to increase sales?

4. How can businesses reduce their operating costs?

5. How can a company improve customer satisfaction?
```

## Output

For every query, the system returns the **top 3 most relevant documents** along with their similarity scores.

Example:

```text
================================================================================
Query: How can a company keep its customers loyal?
================================================================================

1. Similarity Score: 0.xxxx
   Document: Companies can build customer loyalty by offering rewards and loyalty programs.

2. Similarity Score: 0.xxxx
   Document: Customer retention increases when companies provide personalized services and respond quickly to customer complaints.

3. Similarity Score: 0.xxxx
   Document: Businesses can improve customer satisfaction by providing fast and reliable customer support.
```

The exact similarity scores are generated when the program is executed.

## Similarity Scores

A similarity score indicates how closely the meaning of a document matches the meaning of the search query.

A higher score indicates greater semantic similarity.

For example, a document discussing customer loyalty is expected to receive a higher similarity score for a query about retaining customers than an unrelated document about cybersecurity.

## Semantic Search vs. Keyword Search

### Keyword Search

Keyword search mainly looks for matching words.

For example:

```text
customer loyalty
```

A keyword search primarily looks for documents containing the words "customer" and "loyalty".

### Semantic Search

Semantic search focuses on the meaning of the query.

For example:

```text
How can a company keep its customers loyal?
```

can retrieve:

```text
Companies can build customer loyalty by offering rewards and loyalty programs.
```

even though the wording is different.

## Manual vs. Optimized Semantic Search

For small datasets, semantic search can be implemented by directly encoding the corpus and queries and comparing their embeddings.

Sentence Transformers also provides an optimized:

```python
util.semantic_search()
```

function that performs the similarity search and returns the most relevant results.

This project uses `util.semantic_search()` for retrieving the top 3 results.

## Project Structure

```text
Semantic_Search_Project
│
├── business_semantic_search.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows Command Prompt:

```bash
venv\Scripts\activate.bat
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the following command:

```bash
python business_semantic_search.py
```

The program will process all five queries and display the top three semantically similar business documents for each query.

## Requirements

The project requires:

```text
sentence-transformers
torch
```

These dependencies are included in:

```text
requirements.txt
```

## Original Sentence Transformers Example

The original semantic search example was cloned and successfully executed before creating this customized implementation.

The original example is available in the Sentence Transformers repository under:

```text
examples/sentence_transformer/applications/semantic-search/
```

The original example was used as the starting reference for this project.

## Customization

The original semantic search implementation was customized for this assignment by:

* Creating 24 business-related documents.
* Creating 5 different business search queries.
* Returning the top 3 results instead of the original top 5.
* Displaying similarity scores.
* Focusing the search corpus on business-related information.
* Creating a separate implementation for the assignment.

## Assignment Requirements

The following requirements have been completed:

* [x] Clone/download and successfully run the semantic search example.
* [x] Create at least 20 business-related documents.
* [x] Test using at least 5 different search queries.
* [x] Display the top 3 relevant results.
* [x] Display similarity scores.
* [x] Modify the implementation according to the assignment requirements.
* [x] Include installation and execution instructions.
* [ ] Upload the completed project to a personal GitHub repository.

## Conclusion

This project demonstrates how Sentence Transformers can be used to perform semantic search over business-related information.

By representing both queries and documents as embeddings, the system can retrieve relevant documents based on semantic meaning rather than relying only on exact keyword matches.
