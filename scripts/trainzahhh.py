import pandas as pd, os
MODEL_DIR = "/Users/raahimlone/New_Data"
cols = pd.read_parquet(os.path.join(MODEL_DIR, "y.parquet")).columns.tolist()
print(cols)
