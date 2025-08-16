import unittest
import json
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from recommend import recommendation_engine


class ContentEngineTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test data and client."""
        self.app_context = app.app_context()
        self.app_context.push()
        self.client = app.test_client()

        app.config["TESTING"] = True
        app.config["API_TOKEN"] = "test_token"

        self.sample_data = {
            "id": [1, 2, 3, 19, 20],
            "description": [
                "blue shirt",
                "red shirt",
                "blue pants",
                "classic blue shirt",
                "red shorts",
            ],
        }
        self.df = pd.DataFrame(self.sample_data)
        self.sample_data_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
        self.df.to_csv(self.sample_data_path, index=False)

    def tearDown(self):
        """Remove test data file."""
        os.remove(self.sample_data_path)
        self.app_context.pop()

    def test_recommendations(self):
        """Test the recommendation engine from training to prediction."""
        recommendation_engine.train(self.sample_data_path)

        data = {"item": 1, "num": 2}
        headers = {
            "Content-Type": "application/json",
            "X-API-TOKEN": app.config["API_TOKEN"],
        }

        response = self.client.post("/predict", headers=headers, data=json.dumps(data))
        self.assertEqual(response.status_code, 200)

        response_data = json.loads(response.data)

        self.assertIn("query", response_data)
        self.assertEqual(response_data["query"]["id"], 1)

        self.assertIn("recommendations", response_data)
        recommendations = response_data["recommendations"]
        self.assertEqual(len(recommendations), 2)

        self.assertEqual(recommendations[0]["id"], 19)
        self.assertEqual(recommendations[1]["id"], 2)
        self.assertGreater(recommendations[0]["score"], recommendations[1]["score"])


if __name__ == "__main__":
    unittest.main()
