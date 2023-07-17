import numpy as np
from itertools import combinations
import operator
from functools import reduce
import pandas as pd


def __get_df(transactions, k):
    """Get dataframe for k-tuple of items above support"""
    if k == 1:
        # First iteration uses whole set
        df = transactions.copy()
        df.columns = [(v,) for v in df.columns]
        return df
    combos = dict()
    # loop through k-tuple combinations
    for c in combinations(transactions.columns, k):
        # create column where all items in tuple are purchased
        combos[c] = transactions[list(c)].all(axis=1)
    # return df where each column is k-tuple
    return pd.DataFrame(combos)


def __get_items(freq, k):
    """Get set of items which are above support at level k"""
    # concatenate all tuples and return a set to dedupe
    return set(list(reduce(operator.concat, freq.index)))


def __get_freq(transactions, supp_cnt, k):
    """Get frequency itemset at k^th support"""
    df = __get_df(transactions, k)
    # check tuples which are above support, get columns and return sum to show support
    check_sum = df.sum()
    freq = check_sum[check_sum >= supp_cnt]
    if freq.empty:
        return freq, None
    else:
        return freq, __get_items(freq, k)


def apriori(transactions, min_support):
    supp_cnt = min_support*transactions.shape[0]
    tcnt = transactions.shape[0]
    frequency_itemset = pd.Series()
    k = 1
    while True:
        freq, items = __get_freq(transactions, supp_cnt, k)
        frequency_itemset = pd.concat([frequency_itemset, freq])
        if items:
            # update frequency dictionary each step
            # increase level
            k += 1
            # limit transactions to rows which purchased at min k items, only use columns in item list
            transactions = transactions.loc[transactions.sum(axis=1) >= k, list(items)]
        else:
            # if no items stop loop
            break
    itemset_support = pd.DataFrame(frequency_itemset/tcnt, columns=["support"])
    itemset_support['itemsets'] = [frozenset(item) for item in frequency_itemset.index]
    return itemset_support.reset_index(drop=True)

