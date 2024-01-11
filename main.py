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

# Import custom class from utils
from utils.cached_limiter_session import CachedLimiterSession

# Import custom classes from the current dir
from your_module import CachedLimiterSession, QuotesDownloader


# --------------------------------------- MAIN SCRIPT -----------------------------------------------

def main():
    # Set up any configuration or initialization code here

    # Create an instance of QuotesDownloader
    downloader = QuotesDownloader()

    # Create or update the list of tickers
    downloader.create_ticker_list()

    # Download and update stock quotes
    quotes_df = downloader.download_quotes()

    if quotes_df is not None:
        # Process the downloaded quotes DataFrame as needed
        print(quotes_df)


if __name__ == "__main__":
    main()
