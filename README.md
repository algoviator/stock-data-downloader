# Stock Data Downloader

This Python project consists of two main components: `TickersDownloader` and `QuotesDownloader`, each serving different purposes. The project uses various libraries such as `pandas`, `yfinance`, and `requests` for data manipulation, downloading, and API requests. Below are the main functionalities and structures of each component:

## Tickers Downloader (`tickers_downloader.py`)

### Purpose:
- Downloads and updates the list of tickers traded on NYSE, NASDAQ, and AMEX exchanges using the NASDAQ Stock Screener and Yahoo API.
- Downloads history of quotes into files with the ticker symbol as the filename in the data folder.

### Key Files:
- `tickers_downloader.py`: Contains the `TickersDownloader` class responsible for downloading and updating tickers.

### Features:
- Configuration loaded from a YAML file (`config/tickers_downloader.yaml`).
- Custom class for loading configurations (`utils.load_config.LoadConfig`).
- Use of a custom session with rate limiting for API requests (`utils.cached_limiter_session.CachedLimiterSession`).

### Main Steps:
1. **Download Tickers:** Fetches the full list of shares from the NASDAQ Stock Screener API.
2. **Preprocess Data:** Filters, cleans, and updates the columns of the downloaded tickers data.
3. **Add New Data to File:** Appends new data to an existing CSV file, avoiding duplicates.
4. **Update Tickers Data:** Utilizes Yahoo Finance API to update additional data for tickers.

## Quotes Downloader (`quotes_downloader.py`)

### Purpose:
- Downloads and updates historical quotes for each ticker.
- Saves quotes into individual CSV files for each ticker in a specified directory.

### Key Files:
- `quotes_downloader.py`: Contains the `QuotesDownloader` class responsible for downloading and updating quotes.

### Features:
- Configuration loaded from a YAML file (`config/quotes_downloader.yaml`).
- Usage of tickers data for filtering (`config/tickers_downloader.yaml`).
- Custom session with rate limiting for Yahoo Finance API requests.
- Filtering and cleaning historical quotes.

### Main Steps:
1. **Update Quotes:** Continuously downloads and updates historical quotes for tickers based on specified conditions.
2. **Filter and Save Quotes:** Filters and saves historical quotes based on specific conditions, such as minimum average trade value.

## Usage:

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-data-downloader.git
   ```

2. Navigate to the project directory:
   ```bash
   cd stock-data-downloader
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Execution:
1. Run the main script:
   ```bash
   python main.py
   ```

## Customization:

- Customize the configurations in the `config` folder, especially `tickers_downloader.yaml` and `quotes_downloader.yaml`, for adjusting filters, paths, and other settings.

## License:
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Author:
Roman Usoltsev - [LinkedIn](https://www.linkedin.com/in/algoviator/)

© 2024 Roman Usoltsev
