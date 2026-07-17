import pandas as pd
sub = pd.read_csv('data/h100_submission.csv', dtype=str)
for _, r in sub.iterrows():
    print(f"ID: {r['id']}")
    print(f"PRED: {r['pred']}")
    print()
