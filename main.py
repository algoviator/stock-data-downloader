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

# Import custom classes from the current dir
from tickers_downloader import TickersDownloader
#from quotes_downloader import QuotesDownloader

# --------------------------------------- MAIN SCRIPT -----------------------------------------------


def main():

    # Download and update list of tickers traded on NYSE, NASDAQ, and AMEX exchanges
    # using the NASDAQ Stock Screener and Yahoo API.
    ts = TickersDownloader()
    ts.run()


    """
    # quotes_downloader = QuotesDownloader()
    # Create or update the list of tickers
    downloader.create_ticker_list()

    # Download and update stock quotes
    quotes_df = downloader.download_quotes()

    if quotes_df is not None:
        # Process the downloaded quotes DataFrame as needed
        print(quotes_df)
    #"""


if __name__ == "__main__":
    main()
