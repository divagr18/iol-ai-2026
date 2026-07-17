import pandas as pd
import os

# Verify submission looks right
df = pd.read_csv('data/submission.csv', dtype=str)
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nSample predictions:")
for _, r in df.head(5).iterrows():
    print(f"  {r['id']}: {r['pred'][:80]}...")
