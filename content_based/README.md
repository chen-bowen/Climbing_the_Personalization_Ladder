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

### Example Usage

#### Training the Model

To train the recommendation model with the default dataset:

```python
from recommend import RecommendationEngine

# Create an instance of the recommendation engine
recommendation_engine = RecommendationEngine(batch_size=500)

# Train the model with the default data path
recommendation_engine.train("data/amazon_data.parquet")
```

#### Getting Recommendations

To get recommendations for a specific product ID:

```python
# Get top 10 recommendations for a given product ID
recommendations = recommendation_engine.predict("B073CRCDFS", num_recommendations=10)

# Print the recommendations
print(recommendations)
```

#### Running the Application

To run the application and make predictions via the API:

1. **Start the Flask server**:

   ```bash
   python app.py
   ```

2. **Make a prediction request**:

   ```bash
   curl -X POST \
     http://127.0.0.1:5000/predict \
     -H 'Content-Type: application/json' \
     -H 'X-API-TOKEN: your_api_token' \
     -d '{"item": "B073CRCDFS", "num": 10}'
   ```

result:

```
{
  "query": {
    "description": "Product Description Amazon Basics Battle Exercise Training Rope - 30/40/50 Foot Lengths, 1.5/2 Inch Widths From the Manufacturer Amazon Basics",
    "features": "Color Optional: Black Size Optional (1.5\"/2\" Diam.): 30Ft / 40Ft / 50Ft Material: 100% Poly Dacron, Wear Resistant and Durable, 3-Strand Twisted Polydactyl Strong Construction Orange / Yellow Tracking Line, Convenient for Maintenance, 600D Oxford Waterproof Sleeve Protect the Rope from Friction and Fray Heat Shrink Caps on the ends, Heavy and Apprised High Tensile Strength The battle rope is produced under clockwise twist, do not swing counterclockwise",
    "id": "B073CRCDFS",
    "score": null,
    "title": "Amazon Basics Battle Exercise Training Rope - 30/40/50 Foot Lengths, 1.5/2 Inch Widths"
  },
  "recommendations": [
    {
      "description": "Product Description Amazon Basics Battle Exercise Training Rope - 30/40/50 Foot Lengths, 1.5/2 Inch Widths From the Manufacturer Amazon Basics",
      "features": "Color Optional: Black Size Optional (1.5\"/2\" Diam.): 30Ft / 40Ft / 50Ft Material: 100% Poly Dacron, Wear Resistant and Durable, 3-Strand Twisted Polydactyl Strong Construction Orange / Yellow Tracking Line, Convenient for Maintenance, 600D Oxford Waterproof Sleeve Protect the Rope from Friction and Fray Heat Shrink Caps on the ends, Heavy and Apprised High Tensile Strength The battle rope is produced under clockwise twist, do not swing counterclockwise",
      "id": "B073CRCDFS",
      "score": 0.9999999999999996,
      "title": "Amazon Basics Battle Exercise Training Rope - 30/40/50 Foot Lengths, 1.5/2 Inch Widths"
    },
    {
      "description": "OursGym Training Rope Holder conveniently stores more than 50' of rope. It helps store your battle rope in an easy and convenient way so that your rope doesn\u2019t gather dust and get trampled on. WHY CONSIDER OWNING OURSGYM BATTLE ROPE HOLDER? -- Made from high quality steel alloy with extra durable welded joint. -- Round edged hanging bar so that it does not cut or stress battle rope strands. -- Angled stopper helps in easy mounting and dismounting heavy battle ropes. -- Painted with black anti corrosive paint for protection against rust. -- Comes with four holes for fixing screws for a perfect and secured attachment. --- Backed by OursGym, 100% SATISFACTION GUARANTEED, Click \"Add to Cart\" button and keep your battle ropes safe while saving floor space !!!",
      "features": "USES: This Rope Holder conveniently stores more than 50' of rope, Hardware included. BENEFITS: Perfect battle rope storage, keep your training ropes neat and organized.\u00a0 Once done with your workout, simply coil the battle rope on the storage hook to keep it out of the way and perfectly stored for use again. QUALITY & BUILT STURDILY: The incredibly sturdy steel build of the wall storage hook allows you to develop your core strength to the max without worrying whether it can fully support your weight. Enjoy peace of mind and highly productive workout sessions with the OursGym Battle Rope Anchor. DIMENSIONS: 8\" high x 2\" wide x 1/9\" thick x 14\" long holder piece. MONEY BACK GUARANTEE: Not satisfied with your product? Enjoy a 30-day money back guarantee and free replacements for every defective order, backed by OursGym. Click \u201cBuy Now\u201d !!!",
      "id": "B07PD2N2L9",
      "score": 0.23737092151811173,
      "title": "OursGym Training Rope Holder, Exercise Rope Holder, Battle Rope Storage Gym Storage Hook"
    }
  ]
}
```

### Running Tests

```bash
python -m unittest tests/test_recommend.py
```
