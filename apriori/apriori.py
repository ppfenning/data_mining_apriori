import numpy as np
from itertools import combinations


def apriori(transactions, min_support):
    supp_cnt = np.floor(min_support*transactions.shape[0])
    items = transactions.columns
    frequency_itemset = dict()
    k = 1
    while True:
        transactions = transactions.loc[transactions.sum(axis=1) >= k, items]
        if k == 1:
            frequency_itemset.update(
                transactions[transactions.columns[(transactions.sum() >= supp_cnt)]].sum().to_dict()
            )
            items = frequency_itemset.keys()
        else:
            freq = dict()
            for c in combinations(transactions.columns, k):
                check = transactions[list(c)].all(axis=1).sum()
                if check >= supp_cnt:
                    freq[c] = check
            if freq:
                frequency_itemset.update(freq)
                new_items = set()
                for val in freq.keys():
                    for v in val:
                        new_items.add(v)
                items = list(new_items)
            else:
                break
        k += 1
    return frequency_itemset

