# Content Based Recommendation Engine

This project implements a recommendation engine using a Flask API. The engine provides product recommendations based on content similarity, calculated using TF-IDF and cosine similarity on product descriptions. The recommendations are served via a REST API, and the similarity model can be retrained on demand.

## Features

- **Recommendation API**: Get product recommendations for a given item.
- **Train API**: Retrain the recommendation model with new data.
- **Content-Based Similarity**: Uses TF-IDF and cosine similarity to find similar items based on their descriptions.
- **Caching**: Caches item similarities in Redis for fast lookups.
- **Token-Based Authentication**: Protects the API endpoints with a token.

## How it Works

The recommendation engine is built around the concept of finding similar items based on their textual descriptions.

1.  **Training**:

    - Product data is loaded from a CSV file.
    - A TF-IDF (Term Frequency-Inverse Document Frequency) matrix is created from the product descriptions. This converts the text into a matrix of TF-IDF features.
    - The cosine similarity is computed between all pairs of products using their TF-IDF vectors.
    - For each product, the top 100 most similar products are stored in a Redis sorted set.

2.  **Prediction**:
    - When a recommendation is requested for a product, the engine queries Redis to get the pre-computed list of similar items.
    - The top N recommended items are returned, along with their similarity scores.

## Project Structure

- `app.py`: The Flask application that exposes the API endpoints.
- `recommend.py`: Contains the `RecommendationEngine` class, which handles the core logic for training and prediction.
- `data/data.csv`: The default dataset used for training the model.
- `settings.py`: Configuration file for the Flask application.
- `requirements.txt`: A list of the Python packages required to run the project.
- `tests/test_recommend.py`: Unit tests for the recommendation engine.

## API Endpoints

### `/predict`

- **Method**: `POST`
- **Description**: Get recommendations for a given item.
- **Authentication**: Requires `X-API-TOKEN` header.
- **Request Body**:
  ```json
  {
    "item": "product_id",
    "num": 10
  }
  ```
- **Response**:
  ```json
  {
      "query": { ... product details ... },
      "recommendations": [ ... list of recommended products ... ]
  }
  ```

### `/train`

- **Method**: `POST`
- **Description**: Retrain the recommendation model.
- **Authentication**: Requires `X-API-TOKEN` header.
- **Request Body**:
  ```json
  {
    "data-url": "url_to_csv_data"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Success!",
    "success": 1
  }
  ```

## Getting Started

### Prerequisites

- Python 3.6+
- Redis

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd level_1
    ```

2.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Set up the environment variables. You can create a `.env` file or set them directly:
    ```bash
    export FLASK_SECRET="your_secret_key"
    export API_TOKEN="your_api_token"
    export REDIS_URL="redis://localhost:6379"
    ```

### Running the Application

1.  Start the Flask development server:

    ```bash
    python app.py
    ```

2.  Train the model by sending a POST request to the `/train` endpoint:

    ```bash
    curl -X POST \
      http://127.0.0.1:5000/train \
      -H 'Content-Type: application/json' \
      -H 'X-API-TOKEN: your_api_token' \
      -d '{}'
    ```

    This will train the model with the default `data/data.csv` file.

3.  Get recommendations by sending a POST request to the `/predict` endpoint:
    ```bash
    curl -X POST \
      http://127.0.0.1:5000/predict \
      -H 'Content-Type: application/json' \
      -H 'X-API-TOKEN: your_api_token' \
      -d '{"item": "1"}'
    ```

### Running Tests

```bash
python -m unittest tests/test_recommend.py
```
