import orjson
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_data_pipeline(input_path, output_path):
    """
    Loads data from a JSONL file, transforms it, and saves it to a parquet file
    with all fields preserved.

    Args:
        input_path (str): The path to the input JSONL file.
        output_path (str): The path to the output parquet file.
    """
    # Log the start of the data pipeline with input and output paths
    logging.info(
        f"Starting data pipeline. Input: '{input_path}', Output: '{output_path}'"
    )

    data = []  # List to hold processed records
    with open(input_path, "rb") as f:
        for line in f:
            # Parse each line as JSON using orjson
            record = orjson.loads(line)

            if record.get("parent_asin"):
                # Rename parent_asin to id for consistency across records
                record["id"] = record.pop("parent_asin")

                # Iterate over all fields in the record to ensure Parquet compatibility
                for key, value in record.items():
                    if isinstance(value, list):
                        # Convert lists to space-separated strings for storage
                        record[key] = " ".join(map(str, value))
                    elif isinstance(value, dict):
                        # Serialize dictionaries to JSON strings for storage
                        record[key] = orjson.dumps(value).decode("utf-8")

                # Add the processed record to the data list
                data.append(record)

    # Log the number of records loaded and transformed
    logging.info(f"Loaded and transformed {len(data)} records.")

    # Create a DataFrame from the processed data
    df = pd.DataFrame(data)
    # Convert the 'price' column to numeric, coercing errors to NaN
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    # Save the DataFrame to a Parquet file without the index
    df.to_parquet(output_path, index=False)

    # Log the completion of the data pipeline and the output path
    logging.info(f"Data pipeline finished. Output saved to '{output_path}'")


if __name__ == "__main__":
    create_data_pipeline(
        "data/meta_Sports_and_Outdoors.jsonl", "data/amazon_data.parquet"
    )
