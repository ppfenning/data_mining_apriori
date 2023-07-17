import numpy as np
from itertools import combinations
import operator
from functools import reduce

import pandas as pd


def __get_df(transactions, k):
    if k == 1:
        return transactions.copy()
    combos = dict()
    for c in combinations(transactions.columns, k):
        combos[c] = transactions[list(c)].all(axis=1)
    return pd.DataFrame(combos)


def __get_items(freq, k):
    if k == 1:
        return freq.keys()
    return set(list(reduce(operator.concat, freq.keys())))


def __get_freq(transactions, supp_cnt, k):
    df = __get_df(transactions, k)
    freq = df[df.columns[(df.sum() >= supp_cnt)]].sum().to_dict()
    if freq:
        return freq, __get_items(freq, k)
    return freq, None


def apriori(transactions, min_support):
    supp_cnt = np.floor(min_support*transactions.shape[0])
    items = transactions.columns
    frequency_itemset = dict()
    freq = dict()
    k = 1
    while True:
        freq, items = __get_freq(transactions, supp_cnt, k)
        if items:
            frequency_itemset.update(freq)
            k += 1
            transactions = transactions.loc[transactions.sum(axis=1) >= k, list(items)]
        else:
            break
    return frequency_itemset

