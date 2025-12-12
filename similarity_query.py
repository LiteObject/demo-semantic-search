"""
Similarity Query Demo
======================
This module demonstrates various similarity metrics commonly used in semantic search:
- Cosine Similarity
- Dot Product
- Euclidean Distance
- Manhattan Distance

Uses local sentence-transformers models for generating embeddings.
"""

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class SimilarityMetrics:
    """Class containing various similarity/distance metrics for vector comparison."""

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Cosine similarity measures the cosine of the angle between two vectors.
        Range: [-1, 1] where 1 means identical direction, 0 means orthogonal,
        and -1 means opposite direction.

        Formula: cos(θ) = (A · B) / (||A|| * ||B||)
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def dot_product(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate dot product between two vectors.

        Dot product is the sum of element-wise products.
        Higher values indicate more similarity (for normalized vectors).

        Formula: A · B = Σ(ai * bi)
        """
        return np.dot(vec_a, vec_b)

    @staticmethod
    def euclidean_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate Euclidean distance (L2 norm) between two vectors.

        Euclidean distance is the straight-line distance between two points.
        Range: [0, ∞) where 0 means identical vectors.
        Lower values indicate more similarity.

        Formula: ||A - B||₂ = √(Σ(ai - bi)²)
        """
        return float(np.linalg.norm(vec_a - vec_b))

    @staticmethod
    def manhattan_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate Manhattan distance (L1 norm) between two vectors.

        Manhattan distance is the sum of absolute differences.
        Also known as taxicab or city-block distance.
        Range: [0, ∞) where 0 means identical vectors.
        Lower values indicate more similarity.

        Formula: ||A - B||₁ = Σ|ai - bi|
        """
        return float(np.sum(np.abs(vec_a - vec_b)))

    @staticmethod
    def euclidean_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Convert Euclidean distance to a similarity score.

        Uses the formula: similarity = 1 / (1 + distance)
        Range: (0, 1] where 1 means identical vectors.
        """
        distance = SimilarityMetrics.euclidean_distance(vec_a, vec_b)
        return 1.0 / (1.0 + distance)

    @staticmethod
    def manhattan_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Convert Manhattan distance to a similarity score.

        Uses the formula: similarity = 1 / (1 + distance)
        Range: (0, 1] where 1 means identical vectors.
        """
        distance = SimilarityMetrics.manhattan_distance(vec_a, vec_b)
        return 1.0 / (1.0 + distance)


class SimilarityQueryEngine:
    """Engine for performing similarity queries using local embedding models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the similarity query engine with a local model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' - a fast and efficient model.
                       Other options include:
                       - 'all-mpnet-base-v2': Higher quality, slower
                       - 'paraphrase-MiniLM-L6-v2': Good for paraphrase detection
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.metrics = SimilarityMetrics()
        print("Model loaded successfully!")

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True)

    def compare_texts(self, text_a: str, text_b: str) -> dict:
        """
        Compare two texts using all similarity metrics.

        Returns:
            Dictionary containing all similarity scores and distances.
        """
        emb_a = self.get_embedding(text_a)
        emb_b = self.get_embedding(text_b)

        return {
            "cosine_similarity": self.metrics.cosine_similarity(emb_a, emb_b),
            "dot_product": self.metrics.dot_product(emb_a, emb_b),
            "euclidean_distance": self.metrics.euclidean_distance(emb_a, emb_b),
            "euclidean_similarity": self.metrics.euclidean_similarity(emb_a, emb_b),
            "manhattan_distance": self.metrics.manhattan_distance(emb_a, emb_b),
            "manhattan_similarity": self.metrics.manhattan_similarity(emb_a, emb_b),
        }

    def find_most_similar(
        self, query: str, candidates: List[str], metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar candidates to a query using specified metric.

        Args:
            query: The query text to compare against.
            candidates: List of candidate texts to search.
            metric: Similarity metric to use ('cosine', 'dot', 'euclidean', 'manhattan').

        Returns:
            List of (candidate, score) tuples sorted by similarity (highest first).
        """
        query_emb = self.get_embedding(query)
        candidate_embs = self.get_embeddings(candidates)

        scores = []
        for i, cand_emb in enumerate(candidate_embs):
            if metric == "cosine":
                score = self.metrics.cosine_similarity(query_emb, cand_emb)
            elif metric == "dot":
                score = self.metrics.dot_product(query_emb, cand_emb)
            elif metric == "euclidean":
                score = self.metrics.euclidean_similarity(query_emb, cand_emb)
            elif metric == "manhattan":
                score = self.metrics.manhattan_similarity(query_emb, cand_emb)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            scores.append((candidates[i], score))

        # Sort by score descending (higher = more similar)
        return sorted(scores, key=lambda x: x[1], reverse=True)


def demo_similarity_metrics():
    """Demonstrate similarity metrics with example texts."""

    print("=" * 60)
    print("SIMILARITY QUERY DEMO")
    print("=" * 60)

    # Initialize the engine with a local model
    engine = SimilarityQueryEngine(model_name="all-MiniLM-L6-v2")

    # Example texts for comparison
    text_pairs = [
        ("The cat sat on the mat.", "A cat is sitting on a mat."),
        ("The weather is sunny today.", "It's a bright and clear day."),
        (
            "I love programming in Python.",
            "Python is my favorite programming language.",
        ),
        (
            "The cat sat on the mat.",
            "Stock prices rose sharply today.",
        ),  # Unrelated pair
    ]

    print("\n" + "-" * 60)
    print("PAIRWISE TEXT COMPARISON")
    print("-" * 60)

    for text_a, text_b in text_pairs:
        print(f"\nText A: '{text_a}'")
        print(f"Text B: '{text_b}'")
        print()

        results = engine.compare_texts(text_a, text_b)

        print(f"  Cosine Similarity:     {results['cosine_similarity']:.4f}")
        print(f"  Dot Product:           {results['dot_product']:.4f}")
        print(f"  Euclidean Distance:    {results['euclidean_distance']:.4f}")
        print(f"  Euclidean Similarity:  {results['euclidean_similarity']:.4f}")
        print(f"  Manhattan Distance:    {results['manhattan_distance']:.4f}")
        print(f"  Manhattan Similarity:  {results['manhattan_similarity']:.4f}")

    # Demonstrate finding most similar texts
    print("\n" + "-" * 60)
    print("FINDING MOST SIMILAR TEXTS")
    print("-" * 60)

    query = "I want to learn machine learning"
    candidates = [
        "Deep learning is a subset of machine learning",
        "The weather forecast predicts rain tomorrow",
        "Python is great for AI and ML projects",
        "How to get started with artificial intelligence",
        "The stock market closed higher today",
        "Natural language processing tutorial for beginners",
    ]

    print(f"\nQuery: '{query}'")
    print("\nCandidates ranked by similarity:")

    for metric in ["cosine", "dot", "euclidean", "manhattan"]:
        print(f"\n  Using {metric.upper()} metric:")
        results = engine.find_most_similar(query, candidates, metric=metric)
        for rank, (text, score) in enumerate(results, 1):
            print(f"    {rank}. [{score:.4f}] {text}")


if __name__ == "__main__":
    demo_similarity_metrics()
