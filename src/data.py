from pathlib import Path
import pandas as pd

KOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
KOI_PATH = Path("data") / "koi.csv"


def download_koi():
    KOI_PATH.parent.mkdir(exist_ok=True)

    if KOI_PATH.exists():
        print("File already exists, skipping download.")
        return

    df = pd.read_csv(KOI_URL)
    df.to_csv(KOI_PATH, index=False)
    print("Downloaded:", df.shape)


def load_koi():
    download_koi()
    return pd.read_csv(KOI_PATH)


if __name__ == "__main__":
    df = load_koi()
    print("Rows and columns:", df.shape)
    print("Has koi_disposition:", "koi_disposition" in df.columns)