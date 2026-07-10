from app.utils.data_loader import DataLoader


loader = DataLoader()

df = loader.load_dataset()

print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nFirst Five Rows")
print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

print("\nLabel Distribution")
print(df["label"].value_counts())