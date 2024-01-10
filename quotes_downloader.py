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
from datetime import datetime, timedelta

# Related third-party imports
import requests
import numpy as np
import pandas as pd
import yfinance as yf  # price datasets

# Requests rate limiter libs (needed to protect to be blocked)
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter


# ----------------------------------------- CLASSES & FUNCTIONS --------------------------------------

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    # Requests rate limiter libs (needed to protect to be blocked)
    pass


class QuotesDownloader:
    TODAY = datetime.now()
    DATE_FORMAT = '%Y-%m-%d'

    MAX_IPO_YEAR = TODAY.year - 3  # Don't use quotes first 3 years after company's IPO
    TICKER_EXPIRATION = TODAY - timedelta(days=90)  # Expiration date of tickers info. Update needed.
    QUOTES_EXPIRATION = TODAY - timedelta(days=30)  # Expiration date of quotes files. Update needed.

    START_DATE = '1980-01-01'  # The earliest date to start quotes downloading  #datetime(2000, 1, 1)
    MIN_CAP = 500 * 10 ** 6  # minimum of company capitalisation, USD
    MIN_PRICE = 3  # minimum price of one share, USD
    MAX_PRICE = 10 * 10 ** 3  # maximum price of one share, USD

    # Improve columns in the new dataframe made from downloading data
    COLS_DELETE = ['netchange', 'pctchange', 'url']
    COLS_RENAME = {'lastsale': 'price', 'ipoyear': 'ipoYear'}
    COLS_ADD = ['pe', 'beta', 'shortRatio', 'isin']
    COLS_DT_ADD = ['quotesStarted', 'quotesUpdated', 'rowUpdated']
    COLS_TYPES = {'symbol': 'string', 'isin': 'string', 'name': 'string', 'price': 'float', 'volume': 'int',
                  'marketCap': 'int', 'ipoYear': 'int', 'pe': 'float', 'beta': 'float', 'shortRatio': 'float',
                  'country': 'string', 'sector': 'string', 'industry': 'string', 'quotesStarted': 'datetime64[ns]',
                  'quotesUpdated': 'datetime64[ns]', 'rowUpdated': 'datetime64[ns]'}
    COLS_ORDER = ['symbol', 'isin', 'name', 'price', 'volume', 'marketCap', 'ipoYear', 'pe', 'beta',
                  'shortRatio', 'country', 'sector', 'industry', 'quotesStarted', 'quotesUpdated', 'rowUpdated']

    # Web downloader params
    DOWNLOAD_BATCH_SIZE = 30
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

    def __init__(self, tickers_file='data/tickers.csv', quotes_dir='data/quotes/'):

        # CSV file with the list of all tickers
        self.tickers_file = tickers_file

        # Directory of files with tickers quotes
        self.quotes_dir = quotes_dir

        # Session with requests rate limiter (to protect to be blocked)
        self.session = self.initialize_session()

        # """
        # NASDAQ Stock Screener:
        # Download all tickers traded at NYSE, NASDAQ, and AMEX exchanges
        tickers = self.download_ticker_list()
        self.cleaned_tickers = self.clean_ticker_list(tickers)
        self.added_tickers = self.save_ticker_list(self.cleaned_tickers)
        # """

        # YAHOO stocks
        self.update_ticker_data()

    @staticmethod
    def initialize_session():
        """
        Session with requests rate limiter (to protect to be blocked)
        """
        return CachedLimiterSession(
            limiter=Limiter(RequestRate(1, Duration.SECOND * 2)),
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache("yfinance.cache"),
        )

    def download_ticker_list(self):
        """
        Download list of shares from NASDAQ Stock Screener.

        Note:
            Get DataFrame of all tickers traded at NYSE, NASDAQ, and AMEX exchanges.

        :return: DataFrame with headers:
            symbol, name, lastsale, netchange, pctchange, marketCap, country, ipoyear, volume, sector, industry, url
        """
        r = requests.get(
            'https://api.nasdaq.com/api/screener/stocks',
            headers=self.DOWNLOAD_HEADERS,
            params=self.DOWNLOAD_PARAMS)
        data = r.json()['data']

        return pd.DataFrame(data['rows'], columns=data['headers'])

    def refactor_df_columns(self, df):

        # Delete unused and rename other columns
        df = df.drop(columns=self.COLS_DELETE).rename(columns=self.COLS_RENAME)

        # Insert columns for date labels
        df[self.COLS_ADD] = None

        # Insert columns for date labels
        df[self.COLS_DT_ADD] = pd.NaT
        df[self.COLS_DT_ADD] = pd.to_datetime(self.COLS_DT_ADD, format=self.DATE_FORMAT, errors='coerce')

        df = df[self.COLS_ORDER]

        return df

    def clean_ticker_list(self, df):
        """
        Clean and filter the DataFrame of downloaded tickers list
        Note:
            Delete unused columns, change name of other columns. Filter the data by minimum parameters.
        :return: clean DataFrame with headers:
        """

        df = self.refactor_df_columns(df)

        # Remove all Notes/Bonds (not Shares)
        df = df[~df['symbol'].str.contains(r"\.|\^")]

        # Change '/' in Symbol to '-',
        # because on Yahoo finance Simbols looks like "BF-A", not "BF/A"
        df['symbol'] = df['symbol'].apply(lambda x: x.strip().replace('/', '-'))

        # Remove '$' from price before converting to float
        df['price'] = df['price'].str.replace('$', '')

        # Remove decimal from price before converting to int
        df['marketCap'] = df['marketCap'].str.replace('.00', '')

        # Change default values before converting to int
        df['marketCap'] = df['marketCap'].str.strip().replace('', '-1')
        df['ipoYear'] = df['ipoYear'].str.strip().replace('', '-1')

        # Change type of columns
        df = df.astype(self.COLS_TYPES)

        # Filter the data by using predefined values
        df = df[
            ~(df['price'] < self.MIN_PRICE) & ~(df['price'] > self.MAX_PRICE) & ~(df['ipoYear'] >= self.MAX_IPO_YEAR)]

        return df

    def save_ticker_list(self, df):
        """
        """
        try:
            # Read the existing DataFrame from the CSV file
            ex_df = pd.read_csv(self.tickers_file)
        except FileNotFoundError:
            # If the file is not found, create an empty DataFrame
            ex_df = pd.DataFrame(columns=df.columns)

        # Exclude the rows presented in the existing DataFrame
        new_rows = df[~df['symbol'].isin(ex_df['symbol'])]

        # If there are new rows, append them to the end of the old DataFrame
        if not new_rows.empty:
            ex_df = pd.concat([ex_df, new_rows], ignore_index=True)

            # Save the updated DataFrame to the same CSV file
            ex_df.to_csv(self.tickers_file, index=False)

            # Display information about the added rows
            added_rows_count = len(new_rows)
            print(f"Added {added_rows_count} rows to the DataFrame.")
            return new_rows
        else:
            return None

    def update_ticker_data(self):

        # Infinity cycle to download and save tickers
        # until it will be interrapted by condition in string:
        # "if subset.empty: break"
        while True:

            # Read the list of tickers which is not updated or updated long time ago
            # DataFrame from the CSV file
            try:
                df = pd.read_csv(self.tickers_file)
            except FileNotFoundError:
                print(f'File \'{self.tickers_file}\' not found.')
                return None

            # Change type of date columns
            df[self.COLS_DT_ADD] = df[self.COLS_DT_ADD].astype("datetime64[ns]")

            # Выбор строк, где 'rowUpdated' не было, либо старше чем 2 месяца назад
            mask = (
                    (pd.isna(df['rowUpdated'])) |
                    (df['rowUpdated'] < pd.to_datetime(self.TICKER_EXPIRATION))
            )

            subset = df[mask].head(self.DOWNLOAD_BATCH_SIZE)

            # If no rows to update than break from infinity cycle
            if subset.empty:
                break

            symbol_list = subset['symbol'].to_list()
            tickers = yf.Tickers(symbol_list, session=self.session)

            for index, row in subset.iterrows():
                symbol = row['symbol']

                # """
                isin_get = str(tickers.tickers[symbol].isin).strip()
                isin = isin_get if len(isin_get) > 1 else row['isin']

                info = tickers.tickers[symbol].info

                price = round(float(info.get('fiftyDayAverage', row['price'])), 2)
                volume = info.get('averageVolume', row['volume'])
                marketCap = info.get('marketCap', row['marketCap'])

                # Convert date in timestamp format to '%Y-%m-%d'
                timestamp = info.get('firstTradeDateEpochUtc', None)
                ipoYear = pd.to_datetime(timestamp, unit='s').strftime('%Y') if timestamp is not None else \
                row['ipoYear']

                pe = round(float(info.get('trailingPE', row['pe'])), 1)
                beta = round(float(info.get('beta', row['beta'])), 1)
                shortRatio = round(float(info.get('shortRatio', row['shortRatio'])), 2)
                country = info.get('country', row['country'])
                sector = info.get('sector', row['sector'])
                industry = info.get('industry', row['industry'])
                rowUpdated = pd.to_datetime(self.TODAY, format=self.DATE_FORMAT).strftime(self.DATE_FORMAT)

                df.loc[df['symbol'] == symbol, [
                    'isin', 'price', 'volume', 'marketCap', 'ipoYear', 'pe',
                    'beta', 'shortRatio', 'country', 'sector', 'industry', 'rowUpdated']] = [
                    isin, price, volume, marketCap, ipoYear, pe,
                    beta, shortRatio, country, sector, industry, rowUpdated]
                # """

            print(f'Tickers: {symbol_list} were updated')

            # Save the updated DataFrame to the same CSV file
            df.to_csv(self.tickers_file, index=False)
            time.sleep(3)



q = QuotesDownloader()


