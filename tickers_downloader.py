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
# ------------------------------------ IMPORT LIBRARIES ------------------------------------

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

# Import custom class from utils
from utils.load_config import LoadConfig
from utils.cached_limiter_session import CachedLimiterSession

# ---------------------------------- CLASSES & FUNCTIONS -----------------------------------

class TickersDownloader:

    # Date format to show in the date columns of dataframes
    DATE_FORMAT = '%Y-%m-%d'

    # Current day date
    TODAY = datetime.now()

    def __init__(self):
        """
        Initializes the configuration parameters for downloading tickers data and sets up API and file-related settings.

        The function loads configuration from a YAML file and sets up various parameters for filtering tickers data,
        updating dataframe columns, and configuring API requests.
        """
        # Load configuration from the file named like the current and extension '.yaml'
        current_filename = self.get_current_filename()
        self.config = LoadConfig(current_filename).data

        # ----------------------------------------------------------------------------------
        # Filters for downloading tickers data
        # ----------------------------------------------------------------------------------

        # Not download data if company IPO date was less than 3 years ago
        self.MAX_IPO_YEAR = self.TODAY.year - self.config['filter']['years_from_ipo']

        # Expiration date of tickers info. Update ticker info every {x} days
        self.EXPIRATION = self.TODAY - timedelta(days=self.config['filter']['expiration_days'])

        # Minimum/maximum price of one share, USD
        self.MIN_PRICE = self.config['filter']['min_price']
        self.MAX_PRICE = self.config['filter']['max_price']

        # Exclude tickers with stop words in their names
        self.STOP_WORDS = self.config['filter']['stop_words']

        # ----------------------------------------------------------------------------------
        # Dataframe columns updating parameters
        # ----------------------------------------------------------------------------------

        self.COLS_DELETE = self.config['columns']['delete']         # Delete columns from the downloaded dataframe
        self.COLS_RENAME = self.config['columns']['rename']         # Rename columns using standard rules
        self.COLS_ADD = self.config['columns']['add']               # Add new columns with default value None
        self.COLS_DT_ADD = self.config['columns']['add_datetype']   # Add new datetime columns with default value NaT
        self.COLS_TYPES = self.config['columns']['apply_types']     # Columns types
        self.COLS_ORDER = self.config['columns']['new_order']       # Reorder columns before saving to the file

        # ----------------------------------------------------------------------------------
        # NASDAQ and Yahoo API configuration, filepath to save the result of downloading
        # ----------------------------------------------------------------------------------

        # NASDAQ download parameters
        self.DOWNLOAD_BATCH_SIZE = self.config['download']['batch_size']  # Batch size for downloading data at one time
        self.DOWNLOAD_PARAMS = self.config['download']['params']          # API params for downloading
        self.DOWNLOAD_HEADERS = self.config['download']['headers']        # Request headers for downloading

        # Yahoo cached session with the limit for requests frequency
        self.SESSION = CachedLimiterSession.initialize_session()

        # File path to save the result of downloading
        self.PATH = self.config['path']


    @staticmethod
    def get_current_filename():
        """
        Returns:
            str: The current module filename.
        """
        return os.path.splitext(os.path.basename(__file__))[0]

    def create_ticker_list(self):

        # Download from NASDAQ all tickers traded at NYSE, NASDAQ, and AMEX exchanges
        tickers = self.download_ticker_list()

        # Clean and save base ticker list into file
        cleaned_tickers = self.clean_ticker_list(tickers)
        self.added_tickers = self.save_ticker_list(cleaned_tickers)

        # Download necessary info for every ticker from Yahoo
        self.update_ticker_data()


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

    def update_tickers_columns(self, df):

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

        df = self.update_tickers_columns(df)

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

        # Exclude tickers with stop words in their names
        df = df[~df['name'].str.contains('|'.join(self.STOP_WORDS), case=False)]

        # Filter the data by using predefined values
        df = df[
            ~(df['price'] < self.MIN_PRICE) & ~(df['price'] > self.MAX_PRICE) & ~(df['ipoYear'] >= self.MAX_IPO_YEAR)]

        return df

    def save_ticker_list(self, df):
        """
        """
        try:
            # Read the existing DataFrame from the CSV file
            ex_df = pd.read_csv(self.PATH)
        except FileNotFoundError:
            # If the file is not found, create an empty DataFrame
            ex_df = pd.DataFrame(columns=df.columns)

        # Exclude the rows presented in the existing DataFrame
        new_rows = df[~df['symbol'].isin(ex_df['symbol'])]

        # If there are new rows, append them to the end of the old DataFrame
        if not new_rows.empty:
            ex_df = pd.concat([ex_df, new_rows], ignore_index=True)

            # Create a folder (if not exist) to store DataFrame to the file
            os.makedirs(os.path.dirname(self.PATH), exist_ok=True)

            # Save the updated DataFrame to the same file
            ex_df.to_csv(self.PATH, index=False)

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
                df = pd.read_csv(self.PATH)
            except FileNotFoundError:
                print(f'File \'{self.PATH}\' not found.')
                return None

            # Change type of date columns
            df[self.COLS_DT_ADD] = df[self.COLS_DT_ADD].astype("datetime64[ns]")

            # Выбор строк, где 'rowUpdated' не было, либо старше чем 2 месяца назад
            mask = (
                    (pd.isna(df['rowUpdated'])) |
                    (df['rowUpdated'] < pd.to_datetime(self.EXPIRATION))
            )

            subset = df[mask].head(self.DOWNLOAD_BATCH_SIZE)

            # If no rows to update than break from infinity cycle
            if subset.empty: break

            symbol_list = subset['symbol'].to_list()
            tickers = yf.Tickers(symbol_list, session=self.SESSION)

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
            df.to_csv(self.PATH, index=False)
            time.sleep(3)
