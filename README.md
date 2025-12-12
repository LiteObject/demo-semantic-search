# Semantic Search Demo

This repository demonstrates practical implementations of semantic search concepts using embeddings and various distance metrics. It includes examples of how to perform similarity comparisons and nearest-neighbor queries using local embedding models.

## What is Semantic Search?

Traditional search engines rely on keyword matching - they look for exact words or phrases in documents. Semantic search goes deeper by trying to understand the actual meaning and context of both the search query and the documents being searched.

The key insight is that text can be converted into numerical vectors called embeddings, where semantically similar texts end up close to each other in the vector space. This allows us to find relevant documents even when they don't contain the exact same words as the query.

For example, if you search for "how to prepare pasta", a semantic search engine might also return documents about "cooking noodles" or "preparing Italian food", because these concepts are semantically related.

## Core Concepts

### Embeddings

An embedding is a numerical representation of text - a vector of numbers that captures the semantic meaning of that text. Modern embedding models, like those from Sentence Transformers, convert text into these vectors automatically. The important property is that semantically similar texts produce vectors that are close together in the vector space.

### Distance Metrics

Once you have embeddings, you need a way to measure similarity between them. This project demonstrates several common distance metrics:

**Cosine Similarity** - Measures the angle between two vectors. It ranges from -1 to 1, where 1 means the vectors point in the same direction (most similar). This is widely used because it's invariant to vector magnitude.

**Dot Product** - The sum of element-wise products. For normalized vectors, it's equivalent to cosine similarity, but it's faster to compute on large batches.

**Euclidean Distance** - The straight-line distance between two points in the embedding space. Lower values indicate more similarity. Useful when the magnitude of vectors matters.

**Manhattan Distance** - The sum of absolute differences between vector components. Less commonly used than Euclidean, but sometimes useful for high-dimensional data.

### Vector Databases and Indexing

For practical applications with large document collections, you need efficient ways to find the nearest neighbors. Exact methods like brute-force search are simple but slow when you have millions of documents. Approximate methods use techniques like locality-sensitive hashing or random projections to find candidates quickly, then refine the results.

## Files in This Repository

### similarity_query.py

Demonstrates direct similarity comparisons between pairs of texts using all supported distance metrics. This is useful for:
- Understanding how different metrics behave
- Comparing two specific documents or concepts
- Evaluating the quality of your embedding model

Run it to see how different metrics rate the similarity between various text pairs.

### nearest_neighbor_query.py

Implements nearest-neighbor search over a collection of documents. Includes both:
- **Exact NN** - Computes distances to all documents and returns the best matches. Guaranteed to find the true nearest neighbors.
- **Approximate NN** - Uses random projections to reduce dimensionality and speed up search, then re-ranks results using the full embeddings.

This demonstrates how to build a semantic search engine that can handle larger document collections efficiently.

## Getting Started

### Prerequisites

You'll need Python 3.8+ and the following packages:
- numpy - For numerical computations
- sentence-transformers - For generating embeddings with local models

### Installation

Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/Mac
```

Install dependencies:
```bash
pip install numpy sentence-transformers
```

### Running the Demos

Execute the similarity query demo:
```bash
python similarity_query.py
```

Execute the nearest-neighbor query demo:
```bash
python nearest_neighbor_query.py
```

The scripts will download the default embedding model (`all-MiniLM-L6-v2`) on first run, which is a fast and efficient model suitable for most use cases.

## Understanding the Results

When you run the similarity demo, you'll see how different metrics score the same pairs of texts. Notice that:
- Related texts consistently score higher across all metrics
- Different metrics emphasize different aspects of similarity
- Unrelated texts get uniformly low scores

The nearest-neighbor demo shows how the system finds relevant documents from a collection. The benchmark function helps you understand the performance characteristics of different approaches.

## Choosing the Right Metric

- Use **cosine similarity** as your default for most semantic search tasks - it's robust and computationally efficient
- Use **dot product** for very large-scale batch operations where speed is critical
- Use **Euclidean distance** when vector magnitude carries semantic meaning (less common in NLP)
- Use **Manhattan distance** for sparse vectors or when you need a more robust metric for outliers

## Vector Databases and Production Systems

For production use cases with large document collections, consider vector databases like:
- Weaviate - Open-source vector database with semantic search capabilities
- Milvus - High-performance vector similarity search
- Pinecone - Managed vector database service
- FAISS - Facebook's library for efficient similarity search

These systems optimize storage, indexing, and retrieval at scale and often include HNSW or other advanced approximate nearest-neighbor algorithms.

## Further Reading

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Understanding Word Embeddings](https://towardsdatascience.com/word-embeddings-explained-4bef3666e842)
- [Approximate Nearest Neighbors](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor)
- [Weaviate - Vector Database](https://weaviate.io/)
