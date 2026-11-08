from src.data import load_koi

COLUMNS_OF_INTEREST = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
    "koi_insol", "koi_model_snr", "koi_steff", "koi_srad",
]

def fetch_processed_dataset():
    raw_data = load_koi()
    
    # Filter dataset using boolean querying instead of .isin()
    is_valid_disposition = (raw_data["koi_disposition"] == "CONFIRMED") | (raw_data["koi_disposition"] == "FALSE POSITIVE")
    filtered_data = raw_data[is_valid_disposition].copy()
    
    # Map target strings to binary outcomes using a dictionary map
    status_mapping = {"CONFIRMED": 1, "FALSE POSITIVE": 0}
    filtered_data["label"] = filtered_data["koi_disposition"].map(status_mapping)
    
    # Subset required columns
    required_cols = ["kepid", "label"] + COLUMNS_OF_INTEREST
    subset_data = filtered_data[required_cols]
    
    # Track missing values and drop them
    initial_count = len(subset_data)
    clean_data = subset_data.dropna()
    final_count = len(clean_data)
    
    print(f"Rows before dropna: {initial_count}")
    print(f"Rows after dropna:  {final_count}")
    print(f"Removed: {initial_count - final_count}")
    print(clean_data["label"].value_counts())
    
    return clean_data

if __name__ == "__main__":
    fetch_processed_dataset()
