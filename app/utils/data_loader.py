from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Handles loading and merging of the Fake and Real News datasets.
    """

    def __init__(self):
        self.data_path = Path("data/raw")

    def load_dataset(self):
        """
        Loads Fake.csv and True.csv,
        assigns labels, merges them,
        and returns a single DataFrame.
        """

        fake_df = pd.read_csv(self.data_path / "Fake.csv")
        true_df = pd.read_csv(self.data_path / "True.csv")

        fake_df["label"] = 0
        true_df["label"] = 1

        df = pd.concat([fake_df, true_df], ignore_index=True)

        return df