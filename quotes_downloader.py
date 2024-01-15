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

# Loggin reports
import logging

# Standard library imports
import os
import time
from datetime import datetime, timedelta

# Related third-party imports
import pandas as pd
import yfinance as yf  # price datasets

# Import custom class from utils
from utils.load_config import LoadConfig
from utils.cached_limiter_session import CachedLimiterSession

# Prevent warning messages
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------- CLASSES & FUNCTIONS -----------------------------------


class QuotesDownloader:

    def __init__(self):

        # Initialize new logger. Use it to reports about work status into log file
        self.logger = logging.getLogger('quotes')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('logs/quotes_downloader.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Load Quotes configuration from the '.yaml' file
        self.config = LoadConfig('config/quotes_downloader.yaml').data

        # Load Tickers configuration from the '.yaml' file
        self.tickers_config = LoadConfig('config/tickers_downloader.yaml').data

        # ----------------------------------------------------------------------------------
        # Declare constants

        self.DATE_FORMAT = '%Y-%m-%d'  # Date format to show in the date columns of dataframes
        self.TODAY = datetime.now()  # Current day date
        self.MIN_HISTORY_YEARS = self.config['min_history_years']

        # Expiration date. Must update quotes file every {x} days
        self.EXPIRATION = self.TODAY - timedelta(days=self.config['expiration_days'])

        # Get list of types for Ticker DataFrame columns
        self.TICKER_COLS_TYPES = self.tickers_config['columns'][
            'apply_types']  # All columns types to apply to DataFrame
        self.TICKER_COLS_DT_TYPES = self.tickers_config['columns']['add_dt']  # Datetime-type columns list

        # Quotes Dataframe columns
        self.COLS_DELETE = self.config['columns']['delete']  # Delete columns from the downloaded dataframe
        self.COLS_RENAME = self.config['columns']['rename']  # Rename columns using standard rules
        self.COLS_TYPES = self.config['columns']['apply_types']  # All columns types to apply to DataFrame

        # Batch size for downloading data at one time
        self.DOWNLOAD_BATCH_SIZE = self.config['batch_size']

        # Minimum average trade value in millions USD to start saving quotes history
        self.TRADE_VALUE_THRESHOLD = self.config['trade_value_threshold']

        # Yahoo cached session with the limit for requests frequency
        self.SESSION = CachedLimiterSession.initialize_session()

        # File with all tickers (all symbols) of shares
        self.TICKERS_FILE = self.tickers_config['path']

        # Directory to save files with quotes of shares. Filename format: {SYMBOL}.csv
        self.QUOTES_DIR = self.config['path']

    def run(self):

        # Run infinite loop. Stop only if no rows to update found.
        # ----------------------------------------------------------------------------------
        while True:
            # Read the file with tickers data
            try:
                tickers_df = pd.read_csv(self.TICKERS_FILE)
            except Exception as e:
                # Log any exceptions
                self.logger.error(f'File \'{self.TICKERS_FILE}\' not found. Error: {e}', exc_info=True)
                return None

            # Get tickers subset for downloading their quotes. Limited by the batch size.
            subset = self.get_subset_to_update_quotes(tickers_df)

            # Break from infinity cycle if quotes subset for downloading is empty
            if subset.empty:
                # Report on the successful check condition for update
                self.logger.info('All quotes are up to date.')
                return None

            # Download quotes array (all tickers in one) using Yahoo API
            all_quotes = self.download_quotes_from_yahoo(subset)

            # For each ticker update it's quotes file
            for _, current in subset.iterrows():

                # Current ticker ID (symbol)
                current_symbol = current['symbol']

                # Get quotes DataFrame for the current symbol
                current_quotes = self.get_current_quotes_df(all_quotes, current_symbol)

                # Refactoring columns and adjusting column formats
                current_quotes = self.refactor_dataframe_columns(current_quotes)

                # Trim DataFrame. Start data from the date when
                # the average daily trading volume exceeded $100M and no errors found inside.
                current_quotes = self.get_filtered_subset(current_quotes, self.TRADE_VALUE_THRESHOLD)

                # If no quotes pass the filter update status and switch to the next ticker
                if current_quotes.empty:
                    # Update current ticker date label of quotes status
                    tickers_df.loc[tickers_df['symbol'] == current_symbol, ['quotesUpdated']] = [
                        pd.to_datetime(self.TODAY,
                                       format=self.DATE_FORMAT).strftime(self.DATE_FORMAT)]
                    continue

                # Start date of the current quotes subset (for update tickers DataFrame)
                quotesStarted = current_quotes.index[0]

                # If quotes history less than {x} years  switch to the next ticker
                if (self.TODAY.year - quotesStarted.year) < self.MIN_HISTORY_YEARS:
                    continue

                # Save quotes into file
                self.save_quotes_to_file(current_symbol, current_quotes)

                # Update current ticker date label of quotes status
                tickers_df.loc[tickers_df['symbol'] == current_symbol, ['quotesStarted', 'quotesUpdated']] = [
                    quotesStarted,
                    pd.to_datetime(self.TODAY, format=self.DATE_FORMAT).strftime(self.DATE_FORMAT)]

                # Last date of update the row
                # quotesUpdated = pd.to_datetime(self.TODAY, format=self.DATE_FORMAT).strftime(self.DATE_FORMAT)

            # Save the updated DataFrame to the same CSV file
            tickers_df.to_csv(self.TICKERS_FILE, index=False)

            # Report on the successful update of the next batch of tickers
            self.logger.info(
                f'Quotes files of the batch of tickers {subset['symbol'].to_list()} is successfully downloaded.')

            # Pause for 3 seconds before the next data download cycle
            time.sleep(3)

        # Report on the successful download status
        self.logger.info('Download complete.')
        print('Download complete.')

    def get_subset_to_update_quotes(self, df):

        # Update data type for datetime columns to ensure correct operation
        df[self.TICKER_COLS_DT_TYPES] = df[self.TICKER_COLS_DT_TYPES].astype("datetime64[ns]")

        # Create filter to read the list of tickers which is never updated
        # or updated a long time ago
        # !!!!!!!!!!!!!!
        # Mask - оставить только первый фильтр после закачки основных

        mask = ((
                        (pd.isna(df['quotesUpdated'])) |
                        (df['quotesUpdated'] < pd.to_datetime(self.EXPIRATION))
                ) &
                (
                        (df['ipoYear'] < 1990) &
                        (df['price'] > 3) &
                        (df['price'] < 5000) &
                        (df['volume'] * df['price'] > (self.TRADE_VALUE_THRESHOLD * 10 ** 6))
                # Daily average trade volume > $100M
                )
                )

        # Apply filter mask and the batch size to limit count of tickers for one download
        subset = df[mask].head(self.DOWNLOAD_BATCH_SIZE)

        return subset

    def download_quotes_from_yahoo(self, subset):

        try:
            #
            all_quotes = yf.download(
                subset['symbol'].to_list(),
                period="max",
                interval="1d",
                auto_adjust=False,
                rounding=False,
                actions=True,
                group_by='ticker',
                threads=True,
                session=self.SESSION
            )
        except Exception as e:
            # Log any exceptions
            self.logger.error(f'Yfinance download of {subset['symbol'].to_list()} is failed. Error: {e}', exc_info=True)
            return None

        return all_quotes

    @staticmethod
    def get_current_quotes_df(all_quotes, symbol):
        # Если в all_quotes была всего одна акция, то структура массива возвращаемого
        # функцией библиотеки yfinance меняется. Отлавливаем это как ошибку и меняем
        # структуру запроса
        try:
            current_quotes = pd.DataFrame(all_quotes[symbol].dropna())
        except Exception:
            current_quotes = pd.DataFrame(all_quotes.dropna())

        return current_quotes

    def refactor_dataframe_columns(self, df):
        """
        Update columns in the provided DataFrame by removing unused columns, renaming columns,
        and adjusting column formats.

        :param df: DataFrame to be updated.
        :return: Updated DataFrame.
        """
        # Delete unused columns and rename others
        df = df.drop(columns=self.COLS_DELETE).rename(columns=self.COLS_RENAME)

        # Update types of columns
        try:
            df = df.astype(self.COLS_TYPES)
        except Exception as e:
            # sometimes there are no columns 'dividends' and 'splits' in 'df' DataFrame
            # Log any exceptions
            self.logger.error(f'refactor_dataframe_columns(). Error: {e}', exc_info=True)
            # Return empty DataFrame
            return pd.DataFrame(columns=df.columns)

        return df

    @staticmethod
    def get_filtered_subset(subset, trade_threshold):

        # Return empty DataFrame back
        if subset.empty:
            return subset

        # Find the las index (last date) of the rows where one of OHLC <= 0 or
        # daily trading volume < 100,000,000

        # Factor of splits influence to price in past
        splits_factor = subset.loc[subset['splits'] > 0, 'splits'].prod()

        mask = (
                (subset['open'] <= 0) | (subset['high'] <= 0) | (subset['low'] <= 0) | (subset['close'] <= 0) |
                ((subset['volume'] * subset['low'] * splits_factor).rolling(window=14).mean() < (
                            trade_threshold * 10 ** 6))
        )

        not_condition_subset = subset[mask]

        # If all data in condition return full subset back
        if not_condition_subset.empty:
            return subset

        threshold_date = not_condition_subset.index[-1]

        # Exclude from _subset all rows older or equal the last date of excluded rows
        return subset.loc[subset.index > threshold_date]

    def save_quotes_to_file(self, current_symbol, current_quotes):

        # Create a folder (if not exist) to store DataFrame to the file
        os.makedirs(os.path.dirname(self.QUOTES_DIR), exist_ok=True)

        # Save the quotes DataFrame to the file
        current_quotes.to_csv(f'{self.QUOTES_DIR}{current_symbol}.csv')
