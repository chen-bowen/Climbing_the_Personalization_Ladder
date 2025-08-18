# Import necessary libraries
from collections import OrderedDict
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from scipy.sparse import vstack
from tqdm import tqdm
import redis
import logging

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
    SIMILARITY_KEY = "p:smlr:%s"  # Redis key pattern for storing similarities

    def __init__(self, batch_size=500):
        # Initialize the vectorizer with specific parameters
        self.vectorizer = HashingVectorizer(
            n_features=2**16,  # Use a smaller number of features for memory efficiency
            ngram_range=(1, 3),  # Consider unigrams, bigrams, and trigrams
            analyzer="word",  # Analyze words
            stop_words="english",  # Remove English stop words
            alternate_sign=False,  # Do not use alternate sign
        )
        self.batch_size = batch_size  # Set the batch size for processing
        self.redis_client = redis.StrictRedis.from_url("redis://localhost:6379/0")  # Connect to Redis
        self.data = None  # Placeholder for product data
        self.ids = None  # Placeholder for product IDs

    def _load_data(self, data_path="data/amazon_data.parquet"):
        """
        Load product data as pandas DataFrame.
        """
        # Set default data path if None is provided
        if data_path is None:
            data_path = "data/amazon_data.parquet"

        with timer("Loading data"):
            self.data = pd.read_parquet(data_path)[:100000]
            self.ids = self.data["id"].tolist()  # Extract product IDs
            logger.info(f"Loaded {len(self.data)} products.")

    def _vectorize_data(self):
        """
        Combine text features and vectorize them using HashingVectorizer.
        """
        n = len(self.data)  # Number of products
        vector_batches = []  # List to store vectorized batches

        with timer("Vectorizing items"):
            for start in tqdm(range(0, n, self.batch_size), desc="Vectorizing batches"):
                batch = self.data.iloc[start : start + self.batch_size]  # Get a batch of data
                combined_text = (
                    batch["title"].fillna("")
                    + " "
                    + batch["features"].fillna("")
                    + " "
                    + batch["description"].fillna("")
                    + " "
                    + batch["categories"].fillna("")
                )  # Combine text fields
                vecs = self.vectorizer.transform(combined_text)  # Vectorize the combined text
                vector_batches.append(vecs)  # Add the vectorized batch to the list

        # Stack all batches into one sparse matrix
        with timer("Stacking vectors"):
            all_vectors = vstack(vector_batches)  # Stack vectors vertically
        return all_vectors  # Return the stacked vectors

    def _compute_and_cache_similarities(self, all_vectors):
        """
        Compute cosine similarities in batches and cache top 100 in Redis using sparse operations.
        """
        n = all_vectors.shape[0]  # Total number of vectors
        batch_size = self.batch_size  # Batch size for processing

        logger.info("Normalizing vectors...")
        all_vectors_norm = normalize(all_vectors, axis=1)  # Normalize vectors row-wise

        logger.info("Computing and caching similarities in batches...")
        for start in tqdm(range(0, n, batch_size), desc="Processing batches"):
            end = min(start + batch_size, n)  # Determine the end of the batch
            batch_vecs = all_vectors_norm[start:end]  # Get a batch of normalized vectors

            # Sparse dot product (batch_size x n)
            similarities = batch_vecs.dot(all_vectors_norm.T)  # Compute similarities

            for i, idx in enumerate(range(start, end)):
                sim_scores = similarities[i].toarray().ravel()  # Convert similarities to a dense array
                top_indices = np.argpartition(-sim_scores, 100)[:100]  # Get indices of top 100 scores
                top_scores = sim_scores[top_indices]  # Get top 100 scores

                # Store product IDs as strings to handle non-numeric IDs
                # This avoids conversion errors for IDs like 'B0B39ZDT85'
                top_items = {self.ids[j]: float(top_scores[k]) for k, j in enumerate(top_indices)}  # Map top scores to product IDs
                self.redis_client.zadd(self.SIMILARITY_KEY % self.ids[idx], top_items)  # Cache in Redis

    def train(self, data_path="data/amazon_data.parquet"):
        # Load data and compute similarities
        self._load_data(data_path)
        all_vectors = self._vectorize_data()
        self._compute_and_cache_similarities(all_vectors)

    def _format_recommendation(self, item_id, score=None):
        """
        Return product details along with similarity score.
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

    def predict(self, item_id, num_recommendations=10):
        """
        Return top-N recommended items for a given product ID.
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
