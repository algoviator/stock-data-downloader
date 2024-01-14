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

# Loggin reports
import logging

# Standard library imports
import os
import time
from datetime import datetime, timedelta

# Related third-party imports
import requests
import pandas as pd
import yfinance as yf  # price datasets

# Import custom class from utils
from utils.load_config import LoadConfig
from utils.cached_limiter_session import CachedLimiterSession

# ---------------------------------- CLASSES & FUNCTIONS -----------------------------------

class TickersDownloader:

    def __init__(self):
        """
        Initializes the configuration parameters for downloading tickers data and sets up API and file-related settings.

        The function loads configuration from a YAML file and sets up various parameters for filtering tickers data,
        updating dataframe columns, and configuring API requests.
        """
        # Initialize logger. Use it to reports about work status into log file
        logging.basicConfig(
            filename='logs/tickers_downloader.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Load configuration from the '.yaml' file
        self.config = LoadConfig('config/tickers_downloader.yaml').data

        # Date format to show in the date columns of dataframes
        self.DATE_FORMAT = '%Y-%m-%d'

        # Current day date
        self.TODAY = datetime.now()

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

        # Dataframe columns updating parameters
        # ----------------------------------------------------------------------------------

        self.COLS_DELETE = self.config['columns']['delete']  # Delete columns from the downloaded dataframe
        self.COLS_RENAME = self.config['columns']['rename']  # Rename columns using standard rules
        self.COLS_ADD = self.config['columns']['add']  # Add new columns with default value None
        self.COLS_DT_ADD = self.config['columns']['add_dt']  # Add new datetime columns with default value NaT
        self.COLS_TYPES = self.config['columns']['apply_types']  # All columns types to apply to DataFrame
        self.COLS_ORDER = self.config['columns']['new_order']  # Reorder columns before saving to the file

        # NASDAQ and Yahoo API configuration, filepath to save the result of downloading
        # ----------------------------------------------------------------------------------

        # NASDAQ download parameters
        self.URL = self.config['download']['url']  # URL address of NASDAQ stock screener
        self.DOWNLOAD_BATCH_SIZE = self.config['download']['batch_size']  # Batch size for downloading data at one time
        self.DOWNLOAD_PARAMS = self.config['download']['params']  # API params for downloading
        self.DOWNLOAD_HEADERS = self.config['download']['headers']  # Request headers for downloading

        # Yahoo cached session with the limit for requests frequency
        self.SESSION = CachedLimiterSession.initialize_session()

        # File path to save the result of downloading
        self.PATH = self.config['path']

    def run(self):

        # Step 1: Download tickers using NASDAQ API.
        # ----------------------------------------------------------------------------------
        try:
            nasdaq_tickers = self.download_tickers_from_nasdaq()
            # Report on the successful download
            self.logger.info(f'Step 1: Downloaded {len(nasdaq_tickers)} tickers from NASDAQ.')
        except Exception as e:
            # Log any exceptions
            self.logger.error(f'Step 1 failed: {e}', exc_info=True)

        # Step 2: Preprocess downloaded data (restructure, clean, filter)
        # ----------------------------------------------------------------------------------
        try:
            clean_tickers = self.preprocess_downloaded_data(nasdaq_tickers)
            # Report on the successful update
            self.logger.info(f'Step 2: {len(clean_tickers)} tickers passed through the filter.')

        except Exception as e:
            # Log any exceptions
            self.logger.error(f'Step 2 failed: {e}', exc_info=True)

        # Step 3: Add new data to an existing CSV file
        # ----------------------------------------------------------------------------------
        try:
            new_rows = self.add_new_data_to_file(clean_tickers)
            # Report on the successful insert
            self.logger.info(f'Step 3: {len(new_rows)} new rows inserted into Tickers file.')

        except Exception as e:
            # Log any exceptions
            self.logger.error(f'Step 3 failed: {e}', exc_info=True)

        # Step 4: Update ticker data using Yahoo Finance API
        # ----------------------------------------------------------------------------------
        try:
            self.update_tickers_data_using_yahoo()
            # Report on the successful update
            self.logger.info(f'Step 4: All tickers are up to date.')

        except Exception as e:
            # Log any exceptions
            self.logger.error(f'Step 4 failed: {e}', exc_info=True)

    def download_tickers_from_nasdaq(self):
        """
        Downloads the full raw list of shares traded on NYSE, NASDAQ, and AMEX exchanges
        using the NASDAQ Stock Screener API.

        :return: DataFrame
        """
        r = requests.get(self.URL, headers=self.DOWNLOAD_HEADERS, params=self.DOWNLOAD_PARAMS)
        data = r.json()['data']

        return pd.DataFrame(data['rows'], columns=data['headers'])

    def preprocess_downloaded_data(self, df):
        """
        Preprocess the provided DataFrame by applying the following steps:

        1. Update columns by calling update_dataframe_columns.
        2. Update values for successful column type conversion.
        3. Convert columns to standard types.
        4. Remove rows based on specified filters.

        :param df: DataFrame to be preprocessed.
        :return: Preprocessed DataFrame.
        """
        # Step 1: Rename, delete and add new columns into the DataFrame
        # ----------------------------------------------------------------------------------

        # Updating columns and adjusting column formats
        df = self.refactor_dataframe_columns(df)

        # Step 2: Update values for successful converting columns types and do convert
        # ----------------------------------------------------------------------------------

        # Remove '$' from the price field before converting to float
        df['price'] = df['price'].str.replace('$', '')

        # Remove decimal from the price before converting to int
        df['marketCap'] = df['marketCap'].str.replace('.00', '')

        # Change default values before converting to int
        df['marketCap'] = df['marketCap'].str.strip().replace('', '-1')
        df['ipoYear'] = df['ipoYear'].str.strip().replace('', '-1')

        # Change '/' to '-' in the symbol field to meet the Yahoo standard
        df['symbol'] = df['symbol'].apply(lambda x: x.strip().replace('/', '-'))

        # Step 3: Convert columns to standard types
        # ----------------------------------------------------------------------------------
        df = df.astype(self.COLS_TYPES)

        # Step 4: Remove rows from the DataFrame that does not pass the filter
        # ----------------------------------------------------------------------------------

        # Remove all securities that are not shares (Notes/Bonds)
        df = df[~df['symbol'].str.contains(r"\.|\^")]

        # Remove tickers with stop words in their names
        df = df[~df['name'].str.contains('|'.join(self.STOP_WORDS), case=False)]

        # Remove rows not corresponding to financial indicators
        mask = (
                (df['price'] < self.MIN_PRICE) |
                (df['price'] > self.MAX_PRICE) |
                (df['ipoYear'] >= self.MAX_IPO_YEAR)
        )
        df = df[~mask]

        return df

    def refactor_dataframe_columns(self, df):
        """
        Update columns in the provided DataFrame by removing unused columns,
        adding new columns, renaming columns, and adjusting column formats.

        :param df: DataFrame to be updated.
        :return: Updated DataFrame.
        """
        # Delete unused columns and rename others
        df = df.drop(columns=self.COLS_DELETE).rename(columns=self.COLS_RENAME)

        # Add new columns with default values
        df[self.COLS_ADD] = None

        # Add new datetime columns with default values
        df[self.COLS_DT_ADD] = pd.NaT

        # Reorder columns
        df = df[self.COLS_ORDER]

        return df

    def add_new_data_to_file(self, df):
        """
        Add new data to an existing CSV file by reading a DataFrame from the file,
        excluding rows already present, and appending new rows to the existing DataFrame.
        Save the updated DataFrame to the same file.

        :param df: DataFrame with new data to be added.
        :return: DataFrame with the newly added rows or empty DataFrame if no new rows were added.
        """
        try:
            # Read a DataFrame from an existing CSV file
            ex_df = pd.read_csv(self.PATH)
        except FileNotFoundError:
            # If the file is not found, create an empty DataFrame
            ex_df = pd.DataFrame(columns=df.columns)

        # Exclude the rows presented in the existing DataFrame
        new_rows = df[~df['symbol'].isin(ex_df['symbol'])]

        # If there are new rows, append them to the end of the DataFrame from file
        if not new_rows.empty:
            ex_df = pd.concat([ex_df, new_rows], ignore_index=True)

            # Create a folder (if not exist) to store DataFrame to the file
            os.makedirs(os.path.dirname(self.PATH), exist_ok=True)

            # Save the updated DataFrame to the same file
            ex_df.to_csv(self.PATH, index=False)

            return new_rows

        else:
            # Else - return an empty DataFrame
            return pd.DataFrame(columns=df.columns)

    def get_tickers_subset_to_update(self):

        # Read the file with tickers data
        try:
            df = pd.read_csv(self.PATH)
        except FileNotFoundError as e:
            # Log any exceptions
            self.logger.error(f'File \'{self.PATH}\' not found. Error: {e}', exc_info=True)
            return None

        # Update data type for datetime columns to ensure correct operation
        df[self.COLS_DT_ADD] = df[self.COLS_DT_ADD].astype("datetime64[ns]")

        # Create filter to read the list of tickers which is never updated
        # or updated a long time ago
        mask = (
                (pd.isna(df['rowUpdated'])) |
                (df['rowUpdated'] < pd.to_datetime(self.EXPIRATION))
        )

        # Apply filter mask and the batch size to limit count of tickers for one download
        subset = df[mask].head(self.DOWNLOAD_BATCH_SIZE)

        return subset

    def update_tickers_data_using_yahoo(self):
        """
        Update ticker data using Yahoo Finance API.

        This function reads the list of tickers that have not been updated or were updated a long time ago
        from a CSV file, selects a subset of tickers to update, and fetches updated data using Yahoo Finance API.
        It then updates the DataFrame with the new data and saves it back to the same CSV file.

        The process continues in an infinite loop until there are no more tickers to update.

        :return: None
        """
        # Infinite loop while will not execute condition of no rows to update
        while True:

            # Get the limited subset of tickers with expiration condition
            subset = self.get_tickers_subset_to_update()

            # If no rows to update then break from the infinite loop (from the 'while True')
            if subset.empty:
                break

            # Download batch of ticker objects using Yahoo API for the symbols in subset
            subset_symbols = subset['symbol'].to_list()
            subset_tickers = yf.Tickers(subset_symbols, session=self.SESSION)

            # Update each ticker in subset by the data downloaded from Yahoo
            # Use file data by default if the data from Yahoo is missed
            for _, file_data_row in subset.iterrows():

                # Ticker symbol for securities
                symbol = file_data_row['symbol']

                # ISIN - International Securities Identification Number (unique identifier)
                isin_str = str(subset_tickers.tickers[symbol].isin).strip()
                isin = isin_str if len(isin_str) > 1 else file_data_row['isin']

                # Dictionary for the current ticker symbol properties
                ticker_info = subset_tickers.tickers[symbol].info

                # Price from 50-Day Moving Average
                price = round(float(ticker_info.get('fiftyDayAverage', file_data_row['price'])), 2)

                # Volume from 10-Day Moving Average
                volume = ticker_info.get('averageVolume', file_data_row['volume'])

                # Market Cap (intraday)
                marketCap = ticker_info.get('marketCap', file_data_row['marketCap'])

                # Year of company IPO - Initial public offering
                timestamp = ticker_info.get('firstTradeDateEpochUtc', None)
                if timestamp is not None:
                    # Convert date from timestamp format to get the date
                    ipoYear = pd.to_datetime(timestamp, unit='s').strftime('%Y')
                else:
                    ipoYear = file_data_row['ipoYear']

                # Trailing P/E - Price–earnings ratio
                pe = round(float(ticker_info.get('trailingPE', file_data_row['pe'])), 1)

                # Beta - the measure of correlation securities with the market
                beta = round(float(ticker_info.get('beta', file_data_row['beta'])), 1)

                # Short interest ratio - the number of days
                # it takes short sellers on average to cover their positions
                shortRatio = round(float(ticker_info.get('shortRatio', file_data_row['shortRatio'])), 2)

                # Company country
                country = ticker_info.get('country', file_data_row['country'])

                # Sector of economic
                sector = ticker_info.get('sector', file_data_row['sector'])

                # Industrial sector
                industry = ticker_info.get('industry', file_data_row['industry'])

                # Last date of update the row
                rowUpdated = pd.to_datetime(self.TODAY, format=self.DATE_FORMAT).strftime(self.DATE_FORMAT)

                # Update current ticker data by using information downloaded from Yahoo API
                df.loc[df['symbol'] == symbol, [
                    'isin', 'price', 'volume', 'marketCap', 'ipoYear', 'pe',
                    'beta', 'shortRatio', 'country', 'sector', 'industry', 'rowUpdated']
                ] = [
                    isin, price, volume, marketCap, ipoYear, pe,
                    beta, shortRatio, country, sector, industry, rowUpdated
                ]

            # Save the updated DataFrame to the same CSV file
            df.to_csv(self.PATH, index=False)

            # Report on the successful update of the next batch of tickers
            self.logger.info(f'Batch of tickers {subset_symbols} is successfully updated.')

            # Pause for 3 seconds before the next data download cycle
            time.sleep(3)
