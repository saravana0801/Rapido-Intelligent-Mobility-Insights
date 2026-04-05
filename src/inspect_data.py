import pandas as pd
import os

path = "Rapido_dataset"
for f in os.listdir(path):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, f), nrows=100)
        print(f"\n================ {f} ================")
        print(df.info())
        print(df.head(2))
    elif f.endswith('.xlsx'):
        df = pd.read_excel(os.path.join(path, f), nrows=10)
        print(f"\n================ {f} ================")
        print(df.info())
        print(df.head(2))
