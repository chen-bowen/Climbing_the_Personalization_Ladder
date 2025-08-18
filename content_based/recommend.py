# code adapted from: https://blog.untrod.com/2016/06/simple-similar-products-recommendation-engine-in-python.html
from collections import OrderedDict
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import vstack
from tqdm import tqdm
import redis
import logging
from typing import List, Dict, Optional, Union

# Configure logger for information output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import context manager for timing code execution
from contextlib import contextmanager


# Define a timer context manager to measure execution time of code blocks
@contextmanager
def timer(name="Step"):
    start = time.time()
    yield
    end = time.time()
    logger.info(f"{name} took {end - start:.4f} seconds")


# Define the RecommendationEngine class
class RecommendationEngine:
    """
    A recommendation engine that uses text vectorization and cosine similarity to recommend products.

    Attributes:
        SIMILARITY_KEY (str): Redis key pattern for storing similarities.
        vectorizer (HashingVectorizer): Vectorizer for text data.
        batch_size (int): Number of items to process in a batch.
        redis_client (StrictRedis): Redis client for caching similarities.
        data (Optional[pd.DataFrame]): DataFrame containing product data.
        ids (Optional[List[Union[str, int]]]): List of product IDs.
    """

    SIMILARITY_KEY = "p:smlr:%s"  # Redis key pattern for storing similarities

    def __init__(self, batch_size: int = 500) -> None:
        """
        Initializes the RecommendationEngine with a specified batch size.

        Args:
            batch_size (int): The number of items to process in a batch. Default is 500.
        """
        # Initialize the vectorizer with specific parameters
        self.vectorizer = HashingVectorizer(
            n_features=2**16,
            ngram_range=(1, 3),  # Consider unigrams, bigrams, and trigrams
            analyzer="word",  # only look at words
            stop_words="english",
            alternate_sign=False,  # Do not use alternate sign
        )
        self.batch_size = batch_size  # Set the batch size for processing
        self.redis_client = redis.StrictRedis.from_url("redis://localhost:6379/0")  # Connect to Redis

    def _load_data(self, data_path: Optional[str] = "data/amazon_data.parquet") -> None:
        """
        Loads product data from a specified path into a pandas DataFrame.

        Args:
            data_path (Optional[str]): Path to the data file. Default is "data/amazon_data.parquet".

        Raises:
            FileNotFoundError: If the specified data file does not exist.
        """
        # Set default data path if None is provided
        if data_path is None:
            data_path = "data/amazon_data.parquet"

        # Load data, parquet is a columnar format that is efficient for loading large datasets
        # However, the whole dataset is 1.5M rows, which causes a lot of memory issues downstream,
        # so we are only loading the first 100,000 rows for now.
        with timer("Loading data"):
            self.data = pd.read_parquet(data_path)[:100_000]
            self.ids = self.data["id"].tolist()
            logger.info(f"Loaded {len(self.data)} products.")

    def _vectorize_data(self) -> vstack:
        """
        Vectorizes text features of the product data using HashingVectorizer.

        Returns:
            vstack: A sparse matrix of vectorized text features.
        """
        n = len(self.data)  # Number of products
        vector_batches: List = []

        with timer("Vectorizing items"):
            for start in tqdm(range(0, n, self.batch_size), desc="Vectorizing batches"):
                batch = self.data.iloc[start : start + self.batch_size]  # Get a batch of data

                # Combine text features for the batch
                combined_text = (
                    batch["title"].fillna("")
                    + " "
                    + batch["features"].fillna("")
                    + " "
                    + batch["description"].fillna("")
                    + " "
                    + batch["categories"].fillna("")
                )

                # Transform the combined text using the HashingVectorizer
                vecs = self.vectorizer.transform(combined_text)
                vector_batches.append(vecs)

        # Stack all batches into one sparse matrix
        with timer("Stacking vectors"):
            all_vectors = vstack(vector_batches)
        return all_vectors

    def _compute_and_cache_similarities(self, all_vectors: vstack) -> None:
        """
        Computes cosine similarities between product vectors and caches the top 100 similarities in Redis.

        Args:
            all_vectors (vstack): Sparse matrix of vectorized text features.
        """
        n = all_vectors.shape[0]  # Total number of vectors
        batch_size = self.batch_size  # Batch size for processing

        logger.info("Normalizing vectors...")
        # Normalize vectors row-wise
        all_vectors_norm = normalize(all_vectors, axis=1)
        logger.info("Computing and caching similarities in batches...")

        for start in tqdm(range(0, n, batch_size), desc="Processing batches"):
            end = min(start + batch_size, n)
            batch_vecs = all_vectors_norm[start:end]

            # compute similarities
            similarities = batch_vecs.dot(all_vectors_norm.T)

            for i, idx in enumerate(range(start, end)):
                sim_scores = similarities[i].toarray().ravel()  # Convert similarities to a dense array
                top_indices = np.argpartition(-sim_scores, 100)[:100]  # Get indices of top 100 scores
                top_scores = sim_scores[top_indices]  # Get top 100 scores

                # Store product IDs as strings to handle non-numeric IDs
                # This avoids conversion errors for IDs like 'B0B39ZDT85'
                top_items = {self.ids[j]: float(top_scores[k]) for k, j in enumerate(top_indices)}
                self.redis_client.zadd(self.SIMILARITY_KEY % self.ids[idx], top_items)

    def train(self, data_path: Optional[str] = "data/amazon_data.parquet") -> None:
        """
        Trains the recommendation engine by loading data, vectorizing it, and computing similarities.

        Args:
            data_path (Optional[str]): Path to the data file. Default is "data/amazon_data.parquet".
        """
        # Load data and compute similarities
        self._load_data(data_path)
        all_vectors = self._vectorize_data()
        self._compute_and_cache_similarities(all_vectors)

    def _format_recommendation(self, item_id: Union[str, int], score: Optional[float] = None) -> Dict[str, Union[str, float, None]]:
        """
        Formats a recommendation by retrieving product details and including the similarity score.

        Args:
            item_id (Union[str, int]): The ID of the product to format.
            score (Optional[float]): The similarity score to include.

        Returns:
            Dict[str, Union[str, float, None]]: A dictionary containing product details and the similarity score.
        """
        # Check if the item_id exists in the data
        if self.data[self.data["id"] == item_id].empty:
            logger.warning(f"Item ID {item_id} not found in data.")
            return {"id": item_id, "error": "Item not found"}

        row = self.data[self.data["id"] == item_id].iloc[0].to_dict()  # Get product details as a dictionary

        # Format the recommendation to include only title, description, and features
        return OrderedDict(
            {
                "title": row.get("title"),
                "description": row.get("description"),
                "features": row.get("features"),
                "id": item_id,  # Include the ID for reference
                "score": score,  # Include the score if provided
            }
        )

    def predict(
        self, item_id: Union[str, int], num_recommendations: int = 10
    ) -> Dict[str, Union[Dict[str, Union[str, float, None]], List[Dict[str, Union[str, float, None]]]]]:
        """
        Predicts the top-N recommended items for a given product ID.

        Args:
            item_id (Union[str, int]): The ID of the product to get recommendations for.
            num_recommendations (int): The number of recommendations to return. Default is 10.

        Returns:
            Dict[str, Union[Dict[str, Union[str, float, None]], List[Dict[str, Union[str, float, None]]]]]:
            A dictionary containing the query product details and a list of recommended products.
        """
        # Retrieve top-N recommended items from Redis
        recommended_items = self.redis_client.zrevrange(self.SIMILARITY_KEY % item_id, 1, num_recommendations, withscores=True)

        # Exclude the item itself from the recommendations
        # recommended_items = [item for item in recommended_items if item[0].decode("utf-8") != item_id]

        # Format recommendations with product details and scores
        recommendations = [self._format_recommendation(rid.decode("utf-8"), score) for rid, score in recommended_items]

        # Log the formatted recommendations
        logger.info(f"Formatted recommendations: {recommendations}")
        query_item = self._format_recommendation(item_id)  # Get details of the query item
        return {"query": query_item, "recommendations": recommendations}  # Return query and recommendations


recommendation_engine = RecommendationEngine(batch_size=500)  # Create an instance of the recommendation engine


# Main execution block
if __name__ == "__main__":
    recommendation_engine.train("data/amazon_data.parquet")  # Train the engine with data
    recommendation_engine.predict("B073CRCDFS")
