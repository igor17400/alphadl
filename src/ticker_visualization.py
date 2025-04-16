import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from openbb_datamodule import OpenBBDataModule

def plot_bars_with_indicators(
    ohlc, 
    ax, 
    title: str, 
    addplots=[], 
    ref_time="", 
    annotation=False,
):
    last = ohlc.iloc[-1]
    text = (
        "ELECTION DAY\n"
        + f" {last.name.date().strftime('%Y-%m-%d')}\n"
        + f"O: {last['open']:.2f}\n"
        + f"H: {last['high']:.2f}\n"
        + f"L: {last['low']:.2f}\n"
        + f"C: {last['close']:.2f}\n"
        + f"SMA: {last['SMA']:.2f}"
        if "SMA" in ohlc.columns
        else ""
    )

    mpf.plot(
        ohlc.rename(
            {"open": "Open", "high": "High", "low": "Low", "close": "Close"}, axis=1
        ),
        type="candle",
        ax=ax,
        addplot=addplots,
        axtitle=title,
        ylabel="Price",
        vlines=dict(vlines=ref_time, linewidths=20, alpha=0.2) if ref_time else None,
        style="yahoo",
        datetime_format="%Y-%m-%d",
    )

    if annotation:
        ax.annotate(
            text,
            xy=(len(ohlc), last["close"]),
            textcoords="axes fraction",
            xytext=(1.05, 0.4),
            bbox=dict(boxstyle="round", fc="lightgray", alpha=0.8),
            arrowprops=dict(facecolor="black", width=1, alpha=0.5),
        )

def plot_ticker(ticker="TSLA", start_date="2024-08-01", end_date="2024-11-07", election_day="2024-11-06"):
    # Initialize the data module
    data_module = OpenBBDataModule(
        index="SP500",
        train_date=[start_date, end_date],
        val_date=[start_date, end_date],
        test_date=[start_date, end_date],
        interval="1d",
        market="US",
        sequence_length=30,
        batch_size=64,
        cache_dir="./cache",
        use_cache=True,
        provider="yahoo",
        openbb_output_type="dataframe",
        max_prediction_length=5,
        selected_tickers=[ticker]
    )
    
    # Prepare the data
    data_module.prepare_data()
    df = data_module.train_data

    # Filter ticker data
    df_ticker = df[df["ticker"] == ticker]
    df_ticker.set_index("date", inplace=True)
    df_ticker = df_ticker[df_ticker.index >= start_date]
    df_ticker = df_ticker[df_ticker.index <= end_date]
    
    # Get ticker daily data
    df_ticker_daily = df_ticker.resample('D').agg({
        "open": "first", 
        "high": "max", 
        "low": "min", 
        "close": "last", 
        "SMA": "mean"
    }).dropna()
    
    # Get ticker data for election day
    df_ticker_election_day = df_ticker[df_ticker.index.strftime('%Y-%m-%d') == election_day]
    
    fig, (ax_daily, ax_minute) = plt.subplots(1, 2, figsize=(20, 8))

    daily_addplot = mpf.make_addplot(df_ticker_daily["SMA"], panel=0, color="orange", ax=ax_daily)
    minute_addplot = mpf.make_addplot(df_ticker_election_day["EMA"], panel=0, color="orange", ax=ax_minute)
    
    plot_bars_with_indicators(
        df_ticker_daily,
        ax_daily,
        title=f"{ticker} Daily bars with SMA",
        addplots=[daily_addplot],
        annotation=True,
    )

    plot_bars_with_indicators(
        df_ticker_election_day,
        ax_minute,
        title=f"{ticker} Election Day of Trading with EMA",
        addplots=[minute_addplot],
    )
    plt.show()

# Example usage
plot_ticker()