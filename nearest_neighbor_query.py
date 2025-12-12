"""
Nearest-Neighbor (NN) Query Demo
=================================
This module demonstrates nearest-neighbor search using various distance metrics:
- Cosine Similarity
- Dot Product
- Euclidean Distance (L2)
- Manhattan Distance (L1)

Includes implementations using:
- Brute-force search
- Efficient batch processing with NumPy

Uses local sentence-transformers models for generating embeddings.
"""

import time
from typing import List, Tuple, Optional, Literal, cast

import numpy as np
from sentence_transformers import SentenceTransformer

MetricType = Literal["cosine", "dot", "euclidean", "manhattan"]


class DistanceMetrics:
    """Class containing vectorized distance/similarity computations for NN search."""

    @staticmethod
    def cosine_similarity_batch(
        query: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and all embeddings.

        Args:
            query: Query vector of shape (embedding_dim,)
            embeddings: Matrix of shape (n_documents, embedding_dim)

        Returns:
            Array of similarity scores of shape (n_documents,)
        """
        # Normalize vectors
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        emb_norms = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        )

        # Compute dot product with normalized vectors
        return np.dot(emb_norms, query_norm)

    @staticmethod
    def dot_product_batch(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate dot product between query and all embeddings.

        Higher values indicate more similarity.
        """
        return np.dot(embeddings, query)

    @staticmethod
    def euclidean_distance_batch(
        query: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Euclidean distance (L2) between query and all embeddings.

        Lower values indicate more similarity.
        """
        return np.linalg.norm(embeddings - query, axis=1)

    @staticmethod
    def manhattan_distance_batch(
        query: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Manhattan distance (L1) between query and all embeddings.

        Lower values indicate more similarity.
        """
        return np.sum(np.abs(embeddings - query), axis=1)


class NearestNeighborIndex:
    """
    A simple nearest-neighbor index for semantic search.

    This class provides efficient nearest-neighbor search using various
    distance metrics. Documents are embedded and stored for fast retrieval.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the nearest-neighbor index.

        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.metrics = DistanceMetrics()
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[dict] = []
        print("Model loaded successfully!")

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[dict]] = None,
        batch_size: int = 32,
    ) -> None:
        """
        Add documents to the index.

        Args:
            documents: List of document texts to index.
            metadata: Optional list of metadata dictionaries for each document.
            batch_size: Batch size for encoding.
        """
        print(f"Indexing {len(documents)} documents...")
        start_time = time.time()

        # Generate embeddings
        new_embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        # Store documents and embeddings
        self.documents.extend(documents)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Store metadata
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])

        elapsed = time.time() - start_time
        print(f"Indexed {len(documents)} documents in {elapsed:.2f} seconds")
        print(f"Total documents in index: {len(self.documents)}")

    def search(
        self, query: str, k: int = 5, metric: MetricType = "cosine"
    ) -> List[Tuple[str, float, dict]]:
        """
        Search for the k nearest neighbors to the query.

        Args:
            query: The query text.
            k: Number of nearest neighbors to return.
            metric: Distance metric to use.

        Returns:
            List of (document, score, metadata) tuples, sorted by relevance.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Encode query
        query_emb = self.model.encode(query, convert_to_numpy=True)

        # Calculate distances/similarities
        if metric == "cosine":
            scores = self.metrics.cosine_similarity_batch(query_emb, self.embeddings)
            # Higher is better for cosine similarity
            top_indices = np.argsort(scores)[::-1][:k]
        elif metric == "dot":
            scores = self.metrics.dot_product_batch(query_emb, self.embeddings)
            # Higher is better for dot product
            top_indices = np.argsort(scores)[::-1][:k]
        elif metric == "euclidean":
            scores = self.metrics.euclidean_distance_batch(query_emb, self.embeddings)
            # Lower is better for distance, convert to similarity
            top_indices = np.argsort(scores)[:k]
            # Convert distance to similarity score
            scores = 1.0 / (1.0 + scores)
        elif metric == "manhattan":
            scores = self.metrics.manhattan_distance_batch(query_emb, self.embeddings)
            # Lower is better for distance, convert to similarity
            top_indices = np.argsort(scores)[:k]
            # Convert distance to similarity score
            scores = 1.0 / (1.0 + scores)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Build results
        results = []
        for idx in top_indices:
            results.append(
                (self.documents[idx], float(scores[idx]), self.metadata[idx])
            )

        return results

    def search_with_all_metrics(self, query: str, k: int = 5) -> dict:
        """
        Search using all available metrics and compare results.

        Args:
            query: The query text.
            k: Number of nearest neighbors to return per metric.

        Returns:
            Dictionary with results for each metric.
        """
        results = {}
        for metric in ["cosine", "dot", "euclidean", "manhattan"]:
            results[metric] = self.search(query, k=k, metric=cast(MetricType, metric))
        return results

    def clear(self) -> None:
        """Clear all documents from the index."""
        self.documents = []
        self.embeddings = None
        self.metadata = []
        print("Index cleared.")


class ApproximateNearestNeighbor:
    """
    Approximate Nearest Neighbor (ANN) implementation using random projection.

    This is a simple demonstration of how ANN can speed up search by reducing
    the dimensionality of embeddings through random projection (LSH-like approach).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_projections: int = 128):
        """
        Initialize the ANN index.

        Args:
            model_name: Name of the sentence-transformers model.
            n_projections: Number of random projections (reduced dimensions).
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.n_projections = n_projections
        self.projection_matrix: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self.projected_embeddings: Optional[np.ndarray] = None
        self.original_embeddings: Optional[np.ndarray] = None
        print("Model loaded successfully!")

    def _initialize_projections(self, embedding_dim: int) -> None:
        """Initialize random projection matrix."""
        # Random projection for dimensionality reduction
        np.random.seed(42)  # For reproducibility
        self.projection_matrix = np.random.randn(
            embedding_dim, self.n_projections
        ) / np.sqrt(self.n_projections)

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the ANN index."""
        print(f"Indexing {len(documents)} documents with random projection...")

        # Generate embeddings
        embeddings = self.model.encode(documents, convert_to_numpy=True)

        # Initialize projection matrix if needed
        if self.projection_matrix is None:
            self._initialize_projections(embeddings.shape[1])

        # Project embeddings to lower dimension
        assert self.projection_matrix is not None, "Projection matrix not initialized"
        projected = np.dot(embeddings, self.projection_matrix)

        self.documents.extend(documents)

        if self.projected_embeddings is None:
            self.projected_embeddings = projected
            self.original_embeddings = embeddings
        else:
            self.projected_embeddings = np.vstack(
                [self.projected_embeddings, projected]
            )
            assert (
                self.original_embeddings is not None
            ), "Original embeddings should not be None"
            self.original_embeddings = np.vstack([self.original_embeddings, embeddings])

        print(f"Indexed {len(documents)} documents")
        print(
            f"Original dimension: {embeddings.shape[1]}, Projected dimension: {self.n_projections}"
        )

    def _compute_cosine_scores(
        self, embeddings: np.ndarray, query: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity scores for embeddings against query."""
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        emb_norms = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        )
        return np.dot(emb_norms, query_norm)

    def search(
        self, query: str, k: int = 5, n_candidates: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors using approximate search.

        First finds candidates using projected embeddings, then re-ranks
        using original embeddings for better accuracy.

        Args:
            query: Query text.
            k: Number of results to return.
            n_candidates: Number of candidates to consider in first pass.

        Returns:
            List of (document, score) tuples.
        """
        if self.projected_embeddings is None or self.original_embeddings is None:
            return []

        # Encode and project query
        query_emb = self.model.encode(query, convert_to_numpy=True)
        assert self.projection_matrix is not None, "Projection matrix not initialized"
        query_projected = np.dot(query_emb, self.projection_matrix)

        # First pass: find candidates using projected embeddings (faster)
        projected_scores = np.dot(self.projected_embeddings, query_projected)
        n_candidates = min(n_candidates, len(self.documents))
        candidate_indices = np.argsort(projected_scores)[::-1][:n_candidates]

        # Second pass: re-rank candidates using original embeddings (more accurate)
        final_scores = self._compute_cosine_scores(
            self.original_embeddings[candidate_indices], query_emb
        )

        # Sort and return top k
        top_k_indices = np.argsort(final_scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            doc_idx = candidate_indices[idx]
            results.append((self.documents[doc_idx], float(final_scores[idx])))

        return results


def demo_nearest_neighbor():
    """Demonstrate nearest-neighbor search with various metrics."""

    print("=" * 70)
    print("NEAREST-NEIGHBOR (NN) QUERY DEMO")
    print("=" * 70)

    # Sample document collection
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
        "The stock market closed higher today.",
        "Weather forecast predicts sunny skies this weekend.",
        "Python is a popular programming language for data science.",
        "TensorFlow and PyTorch are popular deep learning frameworks.",
        "Supervised learning requires labeled training data.",
        "Unsupervised learning discovers patterns without labels.",
        "Transfer learning uses pre-trained models for new tasks.",
        "The cat sat on the warm windowsill.",
        "Recipe: How to make chocolate chip cookies.",
        "Semantic search understands the meaning of queries.",
    ]

    metadata = [
        {"id": i, "category": "tech" if i < 12 else "other"}
        for i in range(len(documents))
    ]

    # Initialize and build index
    print("\n" + "-" * 70)
    print("BUILDING NEAREST-NEIGHBOR INDEX")
    print("-" * 70)

    nn_index = NearestNeighborIndex(model_name="all-MiniLM-L6-v2")
    nn_index.add_documents(documents, metadata=metadata)

    # Test queries
    queries = [
        "How do neural networks learn?",
        "What is AI?",
        "programming languages for machine learning",
    ]

    # Search with different metrics
    print("\n" + "-" * 70)
    print("NEAREST-NEIGHBOR SEARCH RESULTS")
    print("-" * 70)

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: '{query}'")
        print("=" * 60)

        all_results = nn_index.search_with_all_metrics(query, k=3)

        for metric, results in all_results.items():
            print(f"\n  [{metric.upper()} METRIC]")
            for rank, (doc, score, meta) in enumerate(results, 1):
                print(f"    {rank}. [{score:.4f}] (id={meta['id']}) {doc[:60]}...")

    # Demo Approximate Nearest Neighbor
    print("\n" + "-" * 70)
    print("APPROXIMATE NEAREST-NEIGHBOR (ANN) DEMO")
    print("-" * 70)

    ann_index = ApproximateNearestNeighbor(
        model_name="all-MiniLM-L6-v2", n_projections=64
    )
    ann_index.add_documents(documents)

    query = "deep learning frameworks"
    print(f"\nQuery: '{query}'")
    print("\nANN Results (with random projection):")

    ann_results = ann_index.search(query, k=5)
    for rank, (doc, score) in enumerate(ann_results, 1):
        print(f"  {rank}. [{score:.4f}] {doc}")

    # Compare exact vs approximate
    print("\n" + "-" * 70)
    print("EXACT vs APPROXIMATE NN COMPARISON")
    print("-" * 70)

    exact_results = nn_index.search(query, k=5, metric="cosine")

    print(f"\nQuery: '{query}'")
    print("\nExact NN (brute-force):")
    for rank, (doc, score, _) in enumerate(exact_results, 1):
        print(f"  {rank}. [{score:.4f}] {doc}")

    print("\nApproximate NN (random projection):")
    for rank, (doc, score) in enumerate(ann_results, 1):
        print(f"  {rank}. [{score:.4f}] {doc}")


def benchmark_metrics():
    """Benchmark different distance metrics on a larger dataset."""

    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Generate synthetic documents
    base_texts = [
        "artificial intelligence and machine learning",
        "data science and analytics",
        "software development and programming",
        "cloud computing and infrastructure",
        "cybersecurity and privacy",
    ]

    documents = []
    for i in range(100):
        base = base_texts[i % len(base_texts)]
        documents.append(f"{base} document number {i}")

    print(f"\nBenchmarking with {len(documents)} documents...")

    nn_index = NearestNeighborIndex(model_name="all-MiniLM-L6-v2")
    nn_index.add_documents(documents)

    query = "machine learning applications"
    n_iterations = 10

    print(f"\nRunning {n_iterations} searches per metric...")

    for metric in ["cosine", "dot", "euclidean", "manhattan"]:
        start = time.time()
        for _ in range(n_iterations):
            nn_index.search(query, k=10, metric=cast(MetricType, metric))
        elapsed = time.time() - start
        avg_time = (elapsed / n_iterations) * 1000
        print(f"  {metric.upper():12s}: {avg_time:.2f} ms per search")


if __name__ == "__main__":
    demo_nearest_neighbor()
    benchmark_metrics()
