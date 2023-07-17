# Data Mining Apriori Algorithm

# How to Run

0. Run `poetry install` from the root directory.
   1. Here are instructions for installing [poetry](https://python-poetry.org/docs/#installation) locally.
1. Place transactional csv in the `data` folder.
   1. Use `data/test.csv` as a formatting guide
   2. Each row should represent a single transaction
   3. A transaction includes `n` items which are comma separated
2. In `config/params.env`, set the following parameters:
   1. `TRANSACTIONS`: Path to the transactions file you wish to use (e.g. `data/test.csv`)
   2. `MIN_SUPPORT`: The minimum support percentage 
3. Run `python main.py` from the local directory.

If `check_sum` is zero then `apriori` is working as expected.