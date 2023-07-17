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


def __get_transactions(fpath):
    # use wrong delim so each transaction is a row
    basket = pd.read_csv(TRANSACTIONS, delimiter="|", names=[0])[0].str.split(",")
    # one hot encode, providing full dataset
    mlb = MultiLabelBinarizer()
    return pd.DataFrame(
        mlb.fit_transform(basket),
        columns=mlb.classes_,
        index=basket.index
    ).astype(bool)


def __verify(transactions, min_support):
    # run my apriori
    out = apriori(transactions, min_support)
    # run mlxtend apriori
    ml_out =  ml_apriori(transactions, min_support, use_colnames=True)
    # check sum to verify these are the same
    check_sum = (out.set_index("itemsets") - ml_out.set_index("itemsets")).sum()
    check_sum.index = ["check_sum"]
    return out, ml_out, check_sum


if __name__ == '__main__':
    # verify with checksum
    out, ml_out, chuck_sum = __verify(__get_transactions(TRANSACTIONS), MIN_SUPPORT)
