"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, vol_lookback=60, mom_lookback=20):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.vol_lookback = vol_lookback
        self.mom_lookback = mom_lookback 
        self.portfolio_weights = None
        self.portfolio_returns = None

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        rolling_vol = self.returns[assets].rolling(window=self.vol_lookback).std()
        
        rolling_mom = self.price[assets].pct_change(periods=self.mom_lookback)
        
        inverse_vol = 1.0 / rolling_vol.replace(0, np.nan) 
        
        momentum_adjusted_weights = inverse_vol.where(rolling_mom > 0, other=0)

        sum_momentum_adjusted_weights = momentum_adjusted_weights.sum(axis=1)

        sum_momentum_adjusted_weights.replace(0, 1.0, inplace=True) 

        final_weights = momentum_adjusted_weights.div(sum_momentum_adjusted_weights, axis=0)
        
        start_index = max(self.vol_lookback, self.mom_lookback)
        
        self.portfolio_weights.loc[self.returns.index[start_index:], assets] = final_weights.iloc[start_index:]

        self.portfolio_weights[self.exclude] = 0.0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)
        return self.portfolio_weights

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if self.portfolio_weights is None:
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

        return self.portfolio_returns

    def get_results(self):
        # Ensure portfolio returns are calculated
        if self.portfolio_weights is None:
            self.calculate_weights()

        if self.portfolio_returns is None:
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)