import numpy as np
from itertools import combinations


def apriori(transactions, min_support):
    supp_cnt = np.floor(min_support*transactions.shape[0])
    items = transactions.columns
    frequency_itemset = dict()
    freq = dict()
    k = 1
    while True:
        if k == 1:
            freq = transactions[transactions.columns[(transactions.sum() >= supp_cnt)]].sum().to_dict()
            items = frequency_itemset.keys()
        else:
            freq = dict()
            # TODO: complete this step with map
            for c in combinations(transactions.columns, k):
                check = transactions[list(c)].all(axis=1).sum()
                if check >= supp_cnt:
                    freq[c] = check
            new_items = set()
            for val in freq.keys():
                for v in val:
                    new_items.add(v)
            items = list(new_items)
        if freq:
            frequency_itemset.update(freq)
            k += 1
            transactions = transactions.loc[transactions.sum(axis=1) >= k, items]
        else:
            break
    return frequency_itemset

