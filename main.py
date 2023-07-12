from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv

params = Path("config/params.env")

if params.exists():
    load_dotenv(params)

TRANSACTIONS = Path(os.getenv("TRANSACTIONS"))

if __name__ == '__main__':
    # use wrong delim so each transaction is a row
    transactions = pd.read_csv(TRANSACTIONS, delimiter="|", names=["basket"])
    # break up transaction by item
    transactions["basket"] = transactions["basket"].str.split(",").explode("basket")
    # one hot encode, providing full dataset
    df = pd.get_dummies(transactions["basket"])
