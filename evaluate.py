import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from content_based.recommend import RecommendationEngine


def evaluate(recommender, n=10):
    """
    Evaluates a recommender system.

    Args:
        recommender: An instance of a Recommender class.
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The test data.
        n (int): The number of recommendations to make.

    Returns:
        float: The average precision@n.
    """
    print("Evaluating recommender...")

    # The recommender needs to be trained on the data it will recommend from.
    # We will "train" our recommender on the whole dataset,
    # then "recommend" for items in our test set.
    recommender.train("data/amazon_data.parquet")

    precision_scores = []

    for _, row in test_df.iterrows():
        item_id = row["id"]
        true_category = row["main_category"]

        # We need to make sure the item_id is in the recommender's data
        if item_id not in recommender.data_indexed.index:
            continue

        # Get recommendations for the current item
        recommendations = recommender.predict(item_id, n)

        # The predict method returns a dict with 'query' and 'recommendations'
        recs = recommendations.get("recommendations", [])

        if recs:
            # We need to get the main_category for each recommendation
            rec_ids = [r["id"] for r in recs]
            rec_categories = recommender.data_indexed.loc[rec_ids]["main_category"]

            # Calculate precision@n: the proportion of recommended items that are in the same category
            precision = np.sum(rec_categories == true_category) / n
            precision_scores.append(precision)

    # Calculate the average precision across all test items
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    print("Evaluation complete.")
    return avg_precision


def main():
    """
    Main function to run the evaluation.
    """
    # # Load and prepare data
    # file_path = "data/amazon_data.parquet"
    # df = pd.read_parquet(file_path)
    # # Drop rows with missing essential data
    # df.dropna(subset=["id", "main_category", "description"], inplace=True)

    # # Split data so we have a test set
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize recommender
    recommender = RecommendationEngine()

    # Evaluate the recommender
    avg_precision = evaluate(recommender, train_df, test_df)

    # Print the final evaluation score
    print(f"Average Precision@10: {avg_precision:.4f}")


if __name__ == "__main__":
    main()
