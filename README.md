# Recommender Systems

This repository contains implementations of various recommender system algorithms.

## Implemented Recommender Systems

### 1. Content-Based Filtering

This recommender system suggests items based on their properties. The implementation uses product descriptions and other features to find similar items.

- **Technology**: Python, Flask, Redis, scikit-learn
- **Algorithm**: TF-IDF and Cosine Similarity

For detailed information about the API, setup, and usage, please refer to the specific README for this implementation.

[**>> Go to Content-Based Recommender System**](./content_based/README.md)

## Data

The dataset used for the content-based recommender is from Amazon product data. The raw data is in `JSONL` format and is processed into a `Parquet` file using the script in `content_based/data_pipeline.py`.

- `data/meta_Sports_and_Outdoors.jsonl`: Raw data file.
- `data/amazon_data.parquet`: Processed data file.

## Getting Started

To get started with the content-based recommender system, navigate to the `content_based` directory and follow the instructions in its `README.md` file.

```bash
cd content_based
```

## Future Work

- [ ] Collaborative Filtering
- [ ] Hybrid Models
- [ ] Matrix Factorization techniques (SVD, ALS)
- [ ] Deep Learning based recommenders.
