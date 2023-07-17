from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
from apriori.apriori import apriori
from sklearn.preprocessing import MultiLabelBinarizer

params = Path("config/params.env")

if params.exists():
    load_dotenv(params)

TRANSACTIONS = Path(os.getenv("TRANSACTIONS"))

if __name__ == '__main__':
    # use wrong delim so each transaction is a row
    basket = pd.read_csv(TRANSACTIONS, delimiter="|").iloc[:, 0].str.split(",")
    # one hot encode, providing full dataset
    mlb = MultiLabelBinarizer()
    transactions = pd.DataFrame(
        mlb.fit_transform(basket),
        columns=mlb.classes_,
        index=basket.index
    )
    print(apriori(transactions, .03))
