from app.config.settings import PROCESSED_DATA_DIR
from app.utils.data_loader import DataLoader


def save_processed_dataset():
    loader = DataLoader()

    df = loader.load_dataset()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DATA_DIR / "news_dataset.csv"

    df.to_csv(output_file, index=False)

    print(f"Dataset saved to: {output_file}")


if __name__ == "__main__":
    save_processed_dataset()