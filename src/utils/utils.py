# Reference: https://github.com/andreeaiana/newsreclib/blob/main/newsreclib/utils/utils.py

import warnings
from importlib.util import find_spec
from typing import Callable
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing

    Args:
        cfg: Configuration composed by Hydra.
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```

    Args:
        task_func: Function to be wrapped with the decorator.
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule.

    Args:
        metric_dict: Dictionary of metrics and corresponding values.
        metric_name: Name of the metric to be retrieved.

    Returns:
        Value of the specified metric.
    """

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def historic_resistances(
    df,
    symbol,
    current_date,
    peak_rank_w_pct=0.03,
    strong_peak_prominence_pct=0.05,
    strong_peak_distance=10,
    peak_distance=5,
):
    """
    Find historical resistance levels using peak detection

    Args:
        df: DataFrame with OHLC data
        symbol: Stock symbol
        current_date: Date to calculate resistances for
        peak_rank_w_pct: Percentage window to group nearby peaks
        strong_peak_prominence_pct: Minimum prominence for strong peaks as percentage
        strong_peak_distance: Minimum distance between strong peaks
        peak_distance: Minimum distance between regular peaks

    Returns:
        list: Resistance price levels sorted from strongest to weakest
    """
    # Convert current_date to pandas Timestamp if it isn't already
    current_date = pd.to_datetime(current_date)

    # Get historical data up to current date
    hist_data = df[(df["ticker"] == symbol) & (df.datetime <= current_date)]
    if len(hist_data) < peak_distance:
        return [0]

    # Get price data for peak detection
    prices = hist_data["high"].values

    # Find strong peaks first
    strong_prominence = np.mean(prices) * strong_peak_prominence_pct
    peaks, _ = find_peaks(
        prices, distance=strong_peak_distance, prominence=strong_prominence
    )

    # If not enough strong peaks found, find regular peaks
    if len(peaks) < 2:
        peaks, _ = find_peaks(prices, distance=peak_distance)

    if len(peaks) == 0:
        return [0]

    # Get peak prices
    peak_prices = prices[peaks]

    # Group nearby peaks
    grouped_peaks = []
    current_group = [peak_prices[0]]

    for price in peak_prices[1:]:
        if abs(price - np.mean(current_group)) / price <= peak_rank_w_pct:
            current_group.append(price)
        else:
            grouped_peaks.append(np.mean(current_group))
            current_group = [price]

    grouped_peaks.append(np.mean(current_group))

    # Sort resistance levels from highest to lowest
    resistance_levels = sorted(grouped_peaks, reverse=True)

    return resistance_levels


def historic_supports(
    df,
    symbol,
    current_date,
    trough_rank_w_pct=0.03,
    strong_trough_prominence_pct=0.05,
    strong_trough_distance=10,
    trough_distance=5,
):
    """
    Find historical support levels using trough detection

    Args:
        df: DataFrame with OHLC data
        symbol: Stock symbol
        current_date: Date to calculate supports for
        trough_rank_w_pct: Percentage window to group nearby troughs
        strong_trough_prominence_pct: Minimum prominence for strong troughs as percentage
        strong_trough_distance: Minimum distance between strong troughs
        trough_distance: Minimum distance between regular troughs

    Returns:
        list: Support price levels sorted from lowest to highest
    """
    # Convert current_date to pandas Timestamp if it isn't already
    current_date = pd.to_datetime(current_date)

    # Get historical data up to current date
    hist_data = df[(df["ticker"] == symbol) & (df.datetime <= current_date)]
    if len(hist_data) < trough_distance:
        return [0]

    # Get price data for trough detection
    prices = hist_data["low"].values

    # Find troughs by inverting the price series and finding peaks
    inverted_prices = -prices
    strong_prominence = np.mean(prices) * strong_trough_prominence_pct
    troughs, _ = find_peaks(
        inverted_prices, distance=strong_trough_distance, prominence=strong_prominence
    )

    # If not enough strong troughs found, find regular troughs
    if len(troughs) < 2:
        troughs, _ = find_peaks(inverted_prices, distance=trough_distance)

    if len(troughs) == 0:
        return [0]

    # Get trough prices
    trough_prices = prices[troughs]

    # Group nearby troughs
    grouped_troughs = []
    current_group = [trough_prices[0]]

    for price in trough_prices[1:]:
        if abs(price - np.mean(current_group)) / price <= trough_rank_w_pct:
            current_group.append(price)
        else:
            grouped_troughs.append(np.mean(current_group))
            current_group = [price]

    grouped_troughs.append(np.mean(current_group))

    # Sort support levels from lowest to highest
    support_levels = sorted(grouped_troughs)

    return support_levels


def resample_df(df: pd.DataFrame, resample_period: str, agg_dict: dict) -> pd.DataFrame:
    """Resample DataFrame to a specified period with custom aggregation.

    Args:
        df: DataFrame to resample
        resample_period: Resampling frequency (e.g., 'D' for daily)
        agg_dict: Dictionary specifying aggregation methods for each column

    Returns:
        Resampled DataFrame with additional datetime columns
    """
    # Resample data using groupby and specified aggregation
    resampled_df = df.groupby("ticker").resample(resample_period).agg(agg_dict).dropna()

    # Extract ticker and datetime from multi-index
    resampled_df["ticker"] = resampled_df.index.get_level_values(0)
    resampled_df["datetime"] = resampled_df.index.get_level_values(1)

    # Reset index and add date column
    resampled_df = resampled_df.reset_index(drop=True)
    resampled_df["date_key"] = resampled_df["datetime"].dt.date

    return resampled_df
