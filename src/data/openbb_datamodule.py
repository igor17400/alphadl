import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
import time
from typing import Tuple, Optional
import pickle

import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch
from openbb import obb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.utils import historic_resistances, historic_supports, resample_df

from tqdm import tqdm

tqdm.pandas()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Access the API key
openbb_api_key = os.getenv("OPENBB_API_KEY")

# Login into OpenBB
obb.account.login(pat=openbb_api_key)


class OpenBBDataModule(L.LightningDataModule):
    def __init__(
        self,
        index: str,
        train_date: list,
        val_date: list,
        test_date: list,
        interval: str,
        market: str,
        sequence_length: int,
        batch_size: int,
        cache_dir: str,
        use_cache: bool,
        provider: str,
        openbb_output_type: str,
        max_prediction_length: int,
        load_only_top_losers_and_winners: bool,
        ema_one_minute_bars: int = 9 * 5,
        sma_daily_bars: int = 50,
        selected_tickers: list = None,
        **kwargs,
    ):
        super().__init__()
        self.index = index
        self.train_date = train_date
        self.val_date = val_date
        self.test_date = test_date
        self.interval = interval
        self.market = market
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.provider = provider
        self.openbb_output_type = openbb_output_type
        self.max_prediction_length = max_prediction_length
        self.selected_tickers = selected_tickers
        self.load_only_top_losers_and_winners = load_only_top_losers_and_winners
        # Apply OpenBB settings to ensure output is a dataframe
        obb.user.preferences.output_type = self.openbb_output_type

        # ----- EMA (Exponential Moving Average) is a type of moving
        # average that gives more weight to recent price data.
        # For one-minute bars, a common lookback period
        #       is 45 minutes, which is equivalent to 9 * 5 minutes.
        self.ema_one_minute_bars = ema_one_minute_bars

        # ----- SMA (Simple Moving Average) is a type of moving average
        # that gives equal weight to all price data in the lookback period.
        # For daily bars, a common lookback period is 50 days.
        self.sma_daily_bars = sma_daily_bars

    def calculate_ema(self, df: pd.DataFrame, span: int) -> pd.Series:
        """Calculate Exponential Moving Average for each ticker and date group.

        Args:
            df: DataFrame containing 'ticker', 'date', and 'close' columns
            span: Number of periods for EMA calculation

        Returns:
            Series containing EMA values
        """
        return df.groupby(["ticker", "date"])["close"].transform(
            lambda x: x.ewm(span=span).mean()
        )

    def calculate_close_rank(self, df: pd.DataFrame) -> pd.Series:
        """Calculate close rank (position of close price within day's range).

        Args:
            df: DataFrame containing 'high', 'low', and 'close' columns

        Returns:
            Series containing close rank values (0-1 range)
        """
        return df.apply(
            lambda x: (
                0.5
                if x["high"] == x["low"]
                else (x["close"] - x["low"]) / (x["high"] - x["low"])
            ),
            axis=1,
        )

    def calculate_log_return(self, df: pd.DataFrame) -> pd.Series:
        """Calculate logarithmic returns for each ticker.

        Args:
            df: DataFrame containing 'close' prices

        Returns:
            Series containing log return values
        """
        return df.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))

    def calculate_resistances(
        self,
        df: pd.DataFrame,
        peak_rank_w_pct: float = 0.03,
        strong_peak_prominence_pct: float = 0.05,
        strong_peak_distance: int = 10,
        peak_distance: int = 5,
    ) -> pd.Series:
        """Calculate resistance levels using peak detection.

        Args:
            df: DataFrame containing OHLCV data
            peak_rank_w_pct: Percentage for peak ranking window
            strong_peak_prominence_pct: Percentage for strong peak prominence
            strong_peak_distance: Minimum distance between strong peaks
            peak_distance: Minimum distance between regular peaks

        Returns:
            Series containing resistance levels
        """

        return df.progress_apply(
            lambda x: historic_resistances(
                df,
                x["ticker"],
                x.name,  # This is the date since we're using a datetime index
                peak_rank_w_pct=peak_rank_w_pct,
                strong_peak_prominence_pct=strong_peak_prominence_pct,
                strong_peak_distance=strong_peak_distance,
                peak_distance=peak_distance,
            ),
            axis=1,
        )

    def calculate_supports(
        self,
        df: pd.DataFrame,
        trough_rank_w_pct: float = 0.03,
        strong_trough_prominence_pct: float = 0.05,
        strong_trough_distance: int = 10,
        trough_distance: int = 5,
    ) -> pd.Series:
        """Calculate support levels using trough detection.

        Args:
            df: DataFrame containing OHLCV data
            trough_rank_w_pct: Percentage for trough ranking window
            strong_trough_prominence_pct: Percentage for strong trough prominence
            strong_trough_distance: Minimum distance between strong troughs
            trough_distance: Minimum distance between regular troughs

        Returns:
            Series containing support levels
        """
        return df.progress_apply(
            lambda x: historic_supports(
                df,
                x["ticker"],
                x.name,
                trough_rank_w_pct=trough_rank_w_pct,
                strong_trough_prominence_pct=strong_trough_prominence_pct,
                strong_trough_distance=strong_trough_distance,
                trough_distance=trough_distance,
            ),
            axis=1,
        )

    def get_daily_df(self, minute_df: pd.DataFrame, agg_dict: dict) -> pd.DataFrame:
        """Convert minute-level data to daily data with proper aggregation.

        Args:
            minute_df: DataFrame containing minute-level OHLCV data
            agg_dict: Dictionary specifying aggregation methods for each column

        Returns:
            DataFrame containing daily-level data with properly aligned daily log returns
        """
        # Ensure proper datetime index
        if not isinstance(minute_df.index, pd.DatetimeIndex):
            minute_df = minute_df.set_index("date")

        # Resample using the provided aggregation dictionary
        df_daily = resample_df(minute_df, "D", agg_dict)

        # Calculate SMA on daily data
        df_daily["SMA"] = df_daily.groupby("ticker")["close"].transform(
            lambda x: x.rolling(window=self.sma_daily_bars, min_periods=1).mean()
        )

        return df_daily

    def prepare_data(self):
        """
        Prepares the data for the model by either loading it from a cache file or fetching it from OpenBB.
        """
        # Define the path to the cache directory and tickers subdirectory
        cache_dir = os.path.join(self.cache_dir, self.market, self.interval, self.index)
        tickers_dir = os.path.join(cache_dir, "tickers")
        os.makedirs(tickers_dir, exist_ok=True)  # Ensure the tickers directory exists

        # Define paths for the train, val, and test CSV files (in root cache dir)
        train_file = os.path.join(cache_dir, "train.csv")
        val_file = os.path.join(cache_dir, "val.csv")
        test_file = os.path.join(cache_dir, "test.csv")
        all_file = os.path.join(cache_dir, "ALL.csv")

        # Check if the cache files exist and use them if available
        if self.use_cache and all(
            os.path.exists(f) for f in [train_file, val_file, test_file]
        ):
            logger.info(f"Loading data from cache files in: {cache_dir}")
            self.train_data = pd.read_csv(train_file, index_col=0, parse_dates=True)
            self.val_data = pd.read_csv(val_file, index_col=0, parse_dates=True)
            self.test_data = pd.read_csv(test_file, index_col=0, parse_dates=True)
        else:
            # If cache files do not exist, fetch the data
            logger.info("Fetching data using OpenBB")
            # Fetch the list of constituents for the specified index
            constituents = obb.index.constituents(
                symbol=self.index, provider=self.provider
            )
            sector_map = dict(zip(constituents["symbol"], constituents["sector"]))
            if not os.path.exists(all_file):
                full_data = self._fextch_data(constituents)
            else:
                full_data = pd.read_csv(all_file, index_col=0, parse_dates=True)

            # Ensure the index is a datetime index
            full_data.index = pd.to_datetime(full_data.index)

            # Log the range of the fetched data
            logger.info(
                f"Fetched data from {full_data.index.min()} to {full_data.index.max()}"
            )

            # Add a unified time index
            full_data = self._add_time_idx(full_data)

            # Create daily aggregation before calculating features
            full_data_daily = self.get_daily_df(
                full_data,
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                },
            )

            # Resistance Levels
            logger.info("Calculating resistance levels...")
            full_data_daily["resistances"] = self.calculate_resistances(full_data_daily)

            # Support Levels
            logger.info("Calculating support levels...")
            full_data_daily["supports"] = self.calculate_supports(full_data_daily)

            # Calculate the log_returns
            full_data["log_return"] = self.calculate_log_return(full_data)
            full_data_daily["daily_log_return"] = self.calculate_log_return(
                full_data_daily
            )

            # Extract date from datetime index for merging
            full_data["date_key"] = full_data["date"].dt.date

            # Merge daily log returns into minute data
            full_data = pd.merge(
                full_data,
                full_data_daily[
                    [
                        "ticker",
                        "date_key",
                        "daily_log_return",
                        "SMA",
                        "resistances",
                        "supports",
                    ]
                ],
                on=["ticker", "date_key"],
                how="left",
            )

            # Clean up temporary column
            full_data = full_data.drop("date_key", axis=1)

            # ----- Check if we need to load only top losers and winners
            # This makes processing faster as we have less tickers to process
            if self.load_only_top_losers_and_winners:
                selected_tickers = self.load_top_losers_and_winners(full_data)
                full_data = full_data[full_data["ticker"].isin(selected_tickers)]

            # ----- Calculate additional features
            full_data["day_of_the_week"] = (
                full_data["date"].dt.dayofweek.astype(str).astype("category")
            )
            full_data["hour"] = full_data["date"].dt.hour
            full_data["minute"] = full_data["date"].dt.minute
            full_data["sector"] = full_data["ticker"].map(sector_map)

            # Note:  From now on it might take a long time if we have a lot of tickers
            logger.info("Calculating EMA...")
            full_data["EMA"] = self.calculate_ema(full_data, self.ema_one_minute_bars)

            logger.info("Calculating close rank...")
            full_data["close_rank"] = self.calculate_close_rank(full_data)

            # Save both processed full_data and full_data_daily to CSV
            full_data.to_csv(os.path.join(cache_dir, "processed_full_data.csv"))
            full_data_daily.to_csv(
                os.path.join(cache_dir, "processed_full_data_daily.csv")
            )

            # Convert date strings to datetime objects
            train_start_date, train_end_date = map(
                lambda x: datetime.strptime(x, "%Y-%m-%d"), self.train_date
            )
            val_start_date, val_end_date = map(
                lambda x: datetime.strptime(x, "%Y-%m-%d"), self.val_date
            )
            test_start_date, test_end_date = map(
                lambda x: datetime.strptime(x, "%Y-%m-%d"), self.test_date
            )

            # Split the data
            self.train_data = full_data[
                (full_data.date >= train_start_date)
                & (full_data.date <= train_end_date)
            ]
            self.val_data = full_data[
                (full_data.date >= val_start_date) & (full_data.date <= val_end_date)
            ]
            self.test_data = full_data[
                (full_data.date >= test_start_date) & (full_data.date <= test_end_date)
            ]

            # Save the data to cache
            self.train_data.to_csv(train_file)
            self.val_data.to_csv(val_file)
            self.test_data.to_csv(test_file)

    def _fetch_data(self, constituents):

        tickers = constituents["symbol"].tolist()

        # If selected_tickers is provided, filter the tickers list
        if self.selected_tickers:
            tickers = [ticker for ticker in tickers if ticker in self.selected_tickers]

        # Initialize an empty DataFrame to store the fetched data
        all_data = pd.DataFrame()

        # Set the overall start and end dates
        overall_start_date = datetime.strptime(self.train_date[0], "%Y-%m-%d")
        overall_end_date = datetime.strptime(self.test_date[1], "%Y-%m-%d")

        # Track the number of API calls
        api_call_count = 0
        api_call_limit = 300

        for ticker in tickers:
            current_start_date = overall_start_date
            ticker_data = (
                pd.DataFrame()
            )  # DataFrame to store data for the current ticker
            while current_start_date < overall_end_date:
                # Calculate the end date for the current window
                current_end_date = current_start_date + pd.Timedelta(days=3)
                if current_end_date > overall_end_date:
                    current_end_date = overall_end_date

                # Convert dates to string format for the API call
                start_str = current_start_date.strftime("%Y-%m-%d")
                end_str = current_end_date.strftime("%Y-%m-%d")

                logger.info(f"Fetching data for {ticker} from {start_str} to {end_str}")

                try:
                    # Fetch data for the current window
                    data_chunk = obb.equity.price.historical(
                        symbol=ticker,
                        start_date=start_str,
                        end_date=end_str,
                        interval=self.interval,
                        provider=self.provider,
                    )

                    # Add a ticker column to the data_chunk
                    data_chunk["ticker"] = ticker

                    # Append the fetched data to the ticker_data DataFrame
                    ticker_data = pd.concat([ticker_data, data_chunk])

                except Exception as e:
                    logger.error(f"Failed to fetch data for {ticker}: {e}")
                    break  # Exit the while loop and continue with the next ticker

                # Update the start date for the next window
                current_start_date = current_end_date

                # Increment the API call count
                api_call_count += 1

                # Check if the API call limit is reached
                if api_call_count >= api_call_limit:
                    logger.info("API call limit reached, waiting for 60 seconds...")
                    time.sleep(60)  # Wait for 60 seconds
                    api_call_count = 0  # Reset the API call count

            # Remove duplicates from the ticker data
            ticker_data = ticker_data.drop_duplicates()

            # Save the data for the current ticker in the tickers subdirectory
            ticker_file = os.path.join(
                self.cache_dir,
                self.market,
                self.interval,
                self.index,
                "tickers",
                f"{ticker}.csv",
            )
            ticker_data.to_csv(ticker_file)

            # Append the ticker data to the all_data DataFrame
            all_data = pd.concat([all_data, ticker_data])

        # Remove duplicates from the combined data
        all_data = all_data.drop_duplicates()

        # Save all data to ALL.csv
        all_file = os.path.join(
            self.cache_dir, self.market, self.interval, self.index, "ALL.csv"
        )
        all_data.to_csv(all_file)

        # Log the fetched data to check its content
        logger.info(f"Fetched data head:\n{all_data.head()}")
        logger.info(f"Fetched data tail:\n{all_data.tail()}")

        return all_data

    def _add_time_idx(self, df):
        """
        Adds a time index to the DataFrame for each group (ticker).
        """
        # Ensure the index is a datetime index
        df.index = pd.to_datetime(df.index)

        # Extract month and day from the date
        df["date"] = df.index
        df["month"] = df.index.month
        df["day"] = df.index.day

        def create_time_idx(group):
            # Use pd.factorize to create a continuous index for each symbol's time series
            group["time_idx"] = pd.factorize(group.index)[0]
            return group

        # Apply the time index creation function to each group
        df = df.groupby("ticker").apply(create_time_idx).reset_index(drop=True)

        return df

    def _prepare_sequence_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for model training.

        Args:
            data: DataFrame containing market data

        Returns:
            Tuple of (sequences, targets) where:
            - sequences: Array of shape [num_sequences, sequence_length, num_features]
            - targets: Array of shape [num_sequences, prediction_length, 1]
        """
        sequences = []
        targets = []

        # Select features for input sequences
        feature_columns = [
            "open",
            "high",
            "low",
            "volume",  # Basic price data
            # "EMA",  # Exponential Moving Average
            # "close_rank",  # Position of close price within day's range
            # "daily_log_return",  # Daily log returns
            # "SMA",  # Simple Moving Average
            # "resistances",  # Resistance levels
            # "supports",  # Support levels
            "log_return",  # Log returns
        ]

        # Ensure all required columns exist
        available_features = [col for col in feature_columns if col in data.columns]
        logger.info(f"Using features: {available_features}")

        # Scale the features
        scaler = StandardScaler()
        feature_data = data[available_features].values
        normalized_data = scaler.fit_transform(feature_data)

        # Get log returns for targets and remove NaN values
        target_data = data["log_return"].values.reshape(-1, 1)
        # Remove rows where log_return is NaN
        valid_indices = ~np.isnan(target_data).any(axis=1)
        normalized_data = normalized_data[valid_indices]
        target_data = target_data[valid_indices]
        
        target_scaler = StandardScaler()
        normalized_targets = target_scaler.fit_transform(target_data)

        # Create sequences
        for i in range(
            len(normalized_data) - self.sequence_length - self.max_prediction_length + 1
        ):
            # Input sequence with multiple features
            sequence = normalized_data[i : i + self.sequence_length]
            # Target sequence (log returns)
            target = normalized_targets[
                i
                + self.sequence_length : i
                + self.sequence_length
                + self.max_prediction_length
            ]
            sequences.append(sequence)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def _save_preprocessed_data(self, sequences, targets, stage):
        """Save preprocessed data as pickle files."""
        cache_dir = os.path.join(
            self.cache_dir, self.market, self.interval, self.index, "preprocessed"
        )
        os.makedirs(cache_dir, exist_ok=True)

        # Convert to numpy arrays before saving
        sequences_np = (
            sequences if isinstance(sequences, np.ndarray) else sequences.numpy()
        )
        targets_np = targets if isinstance(targets, np.ndarray) else targets.numpy()

        sequences_file = os.path.join(cache_dir, f"{stage}_sequences.pkl")
        targets_file = os.path.join(cache_dir, f"{stage}_targets.pkl")

        with open(sequences_file, "wb") as f:
            pickle.dump(sequences_np, f)
        with open(targets_file, "wb") as f:
            pickle.dump(targets_np, f)

        logger.info(f"Saved preprocessed {stage} data to {cache_dir}")

    def _load_preprocessed_data(self, stage):
        """Load preprocessed data from pickle files."""
        cache_dir = os.path.join(
            self.cache_dir, self.market, self.interval, self.index, "preprocessed"
        )
        sequences_file = os.path.join(cache_dir, f"{stage}_sequences.pkl")
        targets_file = os.path.join(cache_dir, f"{stage}_targets.pkl")

        if os.path.exists(sequences_file) and os.path.exists(targets_file):
            with open(sequences_file, "rb") as f:
                sequences = pickle.load(f)
            with open(targets_file, "rb") as f:
                targets = pickle.load(f)

            logger.info(f"Loaded preprocessed {stage} data from {cache_dir}")
            return sequences, targets
        return None, None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`"""
        if stage == "fit" or stage is None:
            # Try to load preprocessed data
            train_sequences, train_targets = self._load_preprocessed_data("train")
            val_sequences, val_targets = self._load_preprocessed_data("val")

            # If preprocessed data doesn't exist, process and save it
            if train_sequences is None or val_sequences is None:
                train_sequences, train_targets = self._prepare_sequence_data(
                    self.train_data
                )
                val_sequences, val_targets = self._prepare_sequence_data(self.val_data)

                self._save_preprocessed_data(train_sequences, train_targets, "train")
                self._save_preprocessed_data(val_sequences, val_targets, "val")

            self.data_train = TensorDataset(
                torch.FloatTensor(train_sequences), torch.FloatTensor(train_targets)
            )
            self.data_val = TensorDataset(
                torch.FloatTensor(val_sequences), torch.FloatTensor(val_targets)
            )

        if stage == "test" or stage is None:
            # Try to load preprocessed test data
            test_sequences, test_targets = self._load_preprocessed_data("test")

            # If preprocessed data doesn't exist, process and save it
            if test_sequences is None:
                test_sequences, test_targets = self._prepare_sequence_data(
                    self.test_data
                )
                self._save_preprocessed_data(test_sequences, test_targets, "test")

            self.data_test = TensorDataset(
                torch.FloatTensor(test_sequences), torch.FloatTensor(test_targets)
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def load_top_losers_and_winners(self, df, date="2024-11-06", num=20):
        """
        Returns the top winners and losers for a specific date based on variation/returns.

        Args:
            df: DataFrame containing the data
            date: Target date in 'YYYY-MM-DD' format
            num: Number of top winners and losers to return

        Returns:
            List of selected ticker symbols
        """
        # Ensure index is datetime
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

        # Filter for the specific date (entire trading day)
        date_start = pd.Timestamp(date)
        date_end = date_start + pd.Timedelta(days=1)
        mask = (df.index >= date_start) & (df.index < date_end)
        df_filtered = df[mask]

        # Get top winners (positive variation)
        top_winners = (
            df_filtered.groupby("ticker")["daily_log_return"]
            .mean()
            .sort_values(ascending=False)
            .head(num)
        )

        # Get top losers (negative variation)
        top_losers = (
            df_filtered.groupby("ticker")["daily_log_return"]
            .mean()
            .sort_values(ascending=True)
            .head(num)
        )

        # Combine winners and losers
        selected_tickers = list(set(top_winners.index) | set(top_losers.index))
        logger.info(f"Selected tickers: {selected_tickers}")

        return selected_tickers

    def get_data(self, load_top_losers_and_winners=False):
        """
        Returns the prepared data, optionally filtered for top winners and losers.
        """
        df = self.prepare_data()

        return df
