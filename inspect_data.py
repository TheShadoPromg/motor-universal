import pandas as pd
import sys

try:
    df = pd.read_parquet('data/base/eventos_numericos.parquet')
    print("Columns:", df.columns.tolist())
    print("Last Date:", df['fecha'].max())
    print("Tail:", df.tail(5))
except Exception as e:
    print(e)
