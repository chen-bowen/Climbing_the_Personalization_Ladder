import json
import pandas as pd

from flask import current_app
import redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class RecommendationEngine:
    """
    A recommendation engine that uses TF-IDF and cosine similarity to find similar products.
    """

    SIMILARITY_KEY = "p:smlr:%s"

    def __init__(self):
        self._redis_client = None
        self.tfidf_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=0.0, stop_words="english")
        self.data = None
        self.data_indexed = None

    @property
    def redis_client(self):
        """
        Lazy initialization of the Redis client.
        """
        if self._redis_client is None:
            self._redis_client = redis.StrictRedis.from_url(current_app.config["REDIS_URL"])
        return self._redis_client

    def _load_data(self, data_url):
        """
        Loads the product data from the specified URL or the default local file.
        """
        if data_url:
            self.data = pd.read_parquet(data_url)
        else:
            self.data = pd.read_parquet("data/amazon_data.parquet")
        self.data_indexed = self.data.set_index("id")

    def train(self, data_url) -> None:
        """
        Trains the recommendation engine by building a similarity matrix and caching it in Redis.
        """
        self._load_data(data_url)
        # Create a TF-IDF matrix from the product descriptions
        self.data["combined_features"] = (
            self.data["title"].fillna("")
            + " "
            + self.data["features"].fillna("")
            + " "
            + self.data["description"].fillna("")
            + " "
            + self.data["categories"].fillna("")
        )
        tf_idf_matrix = self.tfidf_vectorizer.fit_transform(self.data["combined_features"])

        # Compute the cosine similarity between all products
        cosine_sim = linear_kernel(tf_idf_matrix, tf_idf_matrix)

        # Cache the top 100 most similar products for each product in Redis
        for idx, row in self.data.iterrows():
            similar_indices = cosine_sim[idx].argsort()[-101:-1]
            similar_items = {int(self.data["id"][i]): float(cosine_sim[idx][i]) for i in similar_indices}
            self.redis_client.zadd(self.SIMILARITY_KEY % row["id"], similar_items)

    def _format_recommendation(self, item_id: int, score: float = None) -> dict:
        """
        Formats a single recommendation item by fetching its data and truncating the description.
        """
        product_data = self.data_indexed.loc[int(item_id)].to_dict()
        product = {"id": int(item_id)}
        if score is not None:
            product["score"] = score
        product.update(product_data)

        if "description" in product and len(product["description"]) > 100:
            product["description"] = product["description"][:100] + "..."
        return product

    def predict(self, product_id: int, n: int = 10) -> dict:
        """
        Returns the top N recommendations for a given product.
        """
        if self.data is None:
            self._load_data(None)
        # Fetch the top N similar items from Redis
        similar_items = self.redis_client.zrevrange(self.SIMILARITY_KEY % product_id, 0, n - 1, withscores=True)

        query = self._format_recommendation(product_id)
        # Format the recommendations
        recommendations = [self._format_recommendation(int(item_id), score) for item_id, score in similar_items]

        return {"query": query, "recommendations": recommendations}


recommendation_engine = RecommendationEngine()
