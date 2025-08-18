import dask
import dask.dataframe as dd
from dask import delayed
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import vstack
import redis
import logging
from contextlib import contextmanager
import time
import faiss
from tqdm import tqdm

# Configure logger for information output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.vectorizer = HashingVectorizer(
            n_features=2**16,
            ngram_range=(1, 3),
            analyzer="word",
            stop_words="english",
            alternate_sign=False,
        )
        self.batch_size = batch_size
        self.redis_client = redis.StrictRedis.from_url("redis://localhost:6379/0")
        self.data = None
        self.ids = None

    def _load_data(self, data_path="data/amazon_data.parquet"):
        logger.info("Loading data...")
        with timer("Loading data"):
            self.data = dd.read_parquet(data_path)
            self.ids = self.data["id"].compute().tolist()
            logger.info(f"Loaded {len(self.data)} products.")

    def _vectorize_data(self):
        logger.info("Vectorizing data...")
        with timer("Vectorizing data"):
            self.data["combined_text"] = (
                self.data["title"] + " " + self.data["features"] + " " + self.data["description"] + " " + self.data["categories"]
            )
            all_vectors = self.vectorizer.fit_transform(self.data["combined_text"].to_bag())
        return all_vectors

    def _compute_and_cache_similarities(self, all_vectors, k=100, batch_size=1000):
        """
        Compute top-K cosine similarities using FAISS and cache them in Redis.
        """
        logger.info("Normalizing vectors...")
        all_vectors_norm = normalize(all_vectors, axis=1).astype("float32")

        n, d = all_vectors_norm.shape
        logger.info(f"Dataset has {n} vectors of dim {d}")

        logger.info("Building FAISS index...")
        index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity (after normalization)
        index.add(all_vectors_norm)

        logger.info(f"Searching top-{k} neighbors for each vector in batches...")
        for start in tqdm(range(0, n, batch_size), desc="FAISS search"):
            batch_vecs = all_vectors_norm[start : start + batch_size]
            # D = similarity scores, I = indices of neighbors
            D, I = index.search(batch_vecs, k)

            pipe = self.redis_client.pipeline(transaction=False)
            for i, idx in enumerate(range(start, min(start + batch_size, n))):
                top_items = {self.ids[j]: float(D[i][m]) for m, j in enumerate(I[i]) if j != idx}
                pipe.zadd(self.SIMILARITY_KEY % self.ids[idx], top_items)
            pipe.execute()

        logger.info("Caching complete.")

    def train(self, data_path="data/amazon_data.parquet"):
        self._load_data(data_path)
        all_vectors = self._vectorize_data()
        self._compute_and_cache_similarities(all_vectors)

    def _format_recommendation(self, item_id, score=None):
        row = self.data[self.data["id"] == item_id].iloc[0].to_dict()
        if score is not None:
            row["score"] = score
        return row

    def predict(self, item_id, num_recommendations=10):
        recommended_items = self.redis_client.zrevrange(self.SIMILARITY_KEY % item_id, 0, num_recommendations - 1, withscores=True)
        recommendations = [self._format_recommendation(int(rid), score) for rid, score in recommended_items]
        query_item = self._format_recommendation(item_id)
        return {"query": query_item, "recommendations": recommendations}


# Main execution block
if __name__ == "__main__":
    engine = RecommendationEngine(batch_size=500)
    engine.train("data/amazon_data.parquet")
