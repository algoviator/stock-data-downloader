# --------------------------------------- HEADER -------------------------------------------
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Roma Usoltsev
# https://www.linkedin.com/in/algoviator/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# --------------------------------------- IMPORT LIBRARIES -------------------------------------------

# Prevent warning messages
import warnings
warnings.filterwarnings('ignore')

# Standard library imports
import os
import time
import datetime as dt

# Related third-party imports
import requests
import numpy as np
import pandas as pd
import yfinance as yf   # price datasets

# --------------------------------------- GLOBAL PARAMETERS -------------------------------------------

START_DATE = '1980-01-01'   # The earliest date to start quotes downloading  #datetime(2000, 1, 1)
MIN_CAP = 500 * 10**6       # minimum of company capitalisation, USD
MIN_PRICE = 3               # minimum price of one share, USD
MAX_PRICE = 10 * 10**3      # maximum price of one share, USD
# Don't trade after company's IPO for 3 years at least.
MAX_IPO_YEAR = dt.datetime.now().year - 3

# Improve columns in the new dataframe made from downloading data
COLS_DELETE = ['netchange', 'pctchange', 'url']
COLS_RENAME = {'lastsale': 'price', 'marketCap': 'market_cap', 'ipoyear': 'ipo'}
COLS_TYPES = {'symbol': 'string', 'name': 'string', 'price': 'float', 'market_cap': 'int', 'country': 'string',
              'ipo': 'int', 'volume': 'int', 'sector': 'string', 'industry': 'string'}

# Web downloader params
DOWNLOAD_PARAMS = (('letter', '0'), ('download', 'true'))
DOWNLOAD_HEADERS = {
    'authority': 'api.nasdaq.com',
    'accept': 'application/json, text/plain, */*',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
    'origin': 'https://www.nasdaq.com',
    'sec-fetch-site': 'same-site',
    'sec-fetch-mode': 'cors',
    'sec-fetch-dest': 'empty',
    'referer': 'https://www.nasdaq.com/',
    'accept-language': 'en-US,en;q=0.9',
}

# ----------------------------------------- CLASSES & FUNCTIONS --------------------------------------
class QuotesDownloader:
    def __init__(self):

        """
        # NASDAQ Stock Screener:
        # Download all tickers traded at NYSE, NASDAQ, and AMEX exchanges
        downloaded_tickers = self.download_ticker_list()
        cleaned_tickers = self.clean_ticker_list(downloaded_tickers)
        self.save_ticker_list(cleaned_tickers)
        #"""

        # YAHOO stocks


    @staticmethod
    def download_ticker_list():
        """
        Download list of shares from NASDAQ Stock Screener.

        Note:
            Get DataFrame of all tickers traded at NYSE, NASDAQ, and AMEX exchanges.

        :return: DataFrame with headers:
            symbol, name, lastsale, netchange, pctchange, marketCap, country, ipoyear, volume, sector, industry, url
        """
        r = requests.get(
            'https://api.nasdaq.com/api/screener/stocks',
            headers=DOWNLOAD_HEADERS,
            params=DOWNLOAD_PARAMS)
        data = r.json()['data']

        return pd.DataFrame(data['rows'], columns=data['headers'])

    @staticmethod
    def clean_ticker_list(df):
        """
        Clean and filter the DataFrame of downloaded tickers list
        Note:
            Delete unused columns, change name of other columns. Filter the data by minimum parameters.
        :return: clean DataFrame with headers:
        """

        # Delete unused and rename other columns
        df = df.drop(columns=COLS_DELETE).rename(columns=COLS_RENAME)

        # Remove all Notes/Bonds (not Shares)
        df = df[~df['symbol'].str.contains(r"\.|\^")]

        # Change '/' in Symbol to '-',
        # because on Yahoo finance Simbols looks like "BF-A", not "BF/A"
        df['symbol'] = df['symbol'].apply(lambda x: x.strip().replace('/', '-'))

        # Remove '$' from price before converting to float
        df['price'] = df['price'].str.replace('$', '')

        # Remove decimal from price before converting to int
        df['market_cap'] = df['market_cap'].str.replace('.00', '')

        # Change default values before converting to int
        df['market_cap'] = df['market_cap'].str.strip().replace('', '-1')
        df['ipo'] = df['ipo'].str.strip().replace('', '-1')

        # Change type of columns
        df = df.astype(COLS_TYPES)

        # Filter the data by using predefined values
        df = df[~(df['price'] < MIN_PRICE) & ~(df['price'] > MAX_PRICE) & ~(df['ipo'] >= MAX_IPO_YEAR)]

        # Updates info for files with quotes and for row in the main table.
        df['quotes_start'], df['quotes_updated'], df['row_updated'] = pd.NaT, pd.NaT, pd.NaT

        return df

    @staticmethod
    def save_ticker_list(df, csv_path='data/tickers.csv'):
        """
        Compares recently downloaded DataFrame with the existing one from the file

        If there are new values, these rows are appended and saved to the same CSV file.

        Parameters:
        - df (pandas.DataFrame): The new DataFrame to compare.
        - csv_path (str): The path to the CSV file containing the existing DataFrame.

        Returns:
        DataFrame with new rows
        """
        # Read the existing DataFrame from the CSV file
        try:
            ex_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            # If the file is not found, create an empty DataFrame
            ex_df = pd.DataFrame(columns=df.columns)

        # Exclude the rows presented in the existing DataFrame
        new_rows = df[~df['symbol'].isin(ex_df['symbol'])]

        # If there are new rows, append them to the end of the old DataFrame
        if not new_rows.empty:
            ex_df = pd.concat([ex_df, new_rows], ignore_index=True)

            # Save the updated DataFrame to the same CSV file
            ex_df.to_csv(csv_path, index=False)

            # Display information about the added rows
            added_rows_count = len(new_rows)
            print(f"Added {added_rows_count} rows to the DataFrame.")
        else:
            print("No new rows were added.")


q = QuotesDownloader()


