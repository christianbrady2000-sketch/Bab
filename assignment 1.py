import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sb
import yfinance as yf
sb.set_theme()

DEFAULT_START = dt.date.isoformat(dt.date.today() - dt.timedelta(365))
DEFAULT_END = dt.date.isoformat(dt.date.today())


class Stock:
    def __init__(self, symbol, start=DEFAULT_START, end=DEFAULT_END):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """Downloads data from yfinance and triggers return calculation."""
        data = yf.download(self.symbol, start=self.start, end=self.end, auto_adjust=True)

        # Flatten multi-level columns if present (yfinance sometimes returns them)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Ensure index is a DatetimeIndex
        data.index = pd.to_datetime(data.index)

        self.calc_returns(data)
        return data

    def calc_returns(self, df):
        """Adds 'Change' and 'Instant_Return' columns to the dataframe."""
        df['Change'] = df['Close'].diff() / df['Close'].shift(1)
        df['Instant_Return'] = np.log(df['Close']).diff().round(4)

    def add_technical_indicators(self, windows=[20, 50]):
        """
        Adds Simple Moving Averages (SMA) for the given windows
        to the internal DataFrame and plots closing price with SMAs.
        """
        for window in windows:
            col_name = f'SMA_{window}'
            self.data[col_name] = self.data['Close'].rolling(window=window).mean()

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=1.5)
        for window in windows:
            ax.plot(self.data.index, self.data[f'SMA_{window}'], label=f'SMA {window}', linewidth=1.2)

        ax.set_title(f'{self.symbol} – Close Price & Moving Averages', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.2f}'))
        plt.tight_layout()
        plt.show()

    def plot_return_dist(self):
        """Plots a histogram of instantaneous returns."""
        fig, ax = plt.subplots(figsize=(10, 5))
        returns = self.data['Instant_Return'].dropna()

        ax.hist(returns, bins=50, edgecolor='white', linewidth=0.5)
        ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {returns.mean():.4f}')
        ax.axvline(returns.median(), color='orange', linestyle='--', linewidth=1.5, label=f'Median: {returns.median():.4f}')

        ax.set_title(f'{self.symbol} – Distribution of Instantaneous Daily Returns', fontsize=14)
        ax.set_xlabel('Instantaneous Return')
        ax.set_ylabel('Frequency')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_performance(self):
        """Plots cumulative growth of $1 investment (percent gain/loss)."""
        cumulative = (1 + self.data['Change'].fillna(0)).cumprod()
        pct_gain = (cumulative - 1) * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, pct_gain, linewidth=1.5, color='steelblue')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        ax.set_title(f'{self.symbol} – Cumulative Performance ({self.start} to {self.end})', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Gain / Loss (%)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.tight_layout()
        plt.show()


def main():
    # Instantiate a test object
    aapl = Stock("AAPL")

    # Access the data attribute
    print(aapl.data.head())

    # Generate the two required plots
    aapl.plot_return_dist()
    aapl.plot_performance()

    # Bonus: technical indicators
    aapl.add_technical_indicators()


if __name__ == "__main__":
    main()