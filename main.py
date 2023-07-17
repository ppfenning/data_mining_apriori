from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
from apriori.apriori import apriori
from mlxtend.frequent_patterns import apriori as ml_apriori
from sklearn.preprocessing import MultiLabelBinarizer

params = Path("config/params.env")

if params.exists():
    load_dotenv(params)

TRANSACTIONS = Path(os.getenv("TRANSACTIONS"))
MIN_SUPPORT = float(os.getenv("MIN_SUPPORT"))

if __name__ == '__main__':
    # use wrong delim so each transaction is a row
    min_support = 0.03
    basket = pd.read_csv(TRANSACTIONS, delimiter="|", names=[0])[0].str.split(",")
    # one hot encode, providing full dataset
    mlb = MultiLabelBinarizer()
    transactions = pd.DataFrame(
        mlb.fit_transform(basket),
        columns=mlb.classes_,
        index=basket.index
    )
    mine = apriori(transactions, MIN_SUPPORT)
    theirs = ml_apriori(transactions, MIN_SUPPORT, use_colnames=True)
