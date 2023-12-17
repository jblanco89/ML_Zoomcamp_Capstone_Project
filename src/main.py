"""
Execute Time Series Forecasting Script from command line

This script serves as a command-line interface (CLI) for executing the main forecasting script.
It reads a time series CSV file specified as an argument, processes the data using the main script,
and generates forecasts for a subset of stores. The default settings include saving MAPE results and plotting figures.

Usage:
python main.py path/to/all_data_ts.csv

Example:
python src/main.py src/Features/all_data_ts.csv

Dependencies:
- argparse
- pandas
- test module (containing run_app function)
"""


import argparse
import pandas as pd
from test import run_app

#Note: if you want to generate more results just change 'ds' parameter in run_app, 
# for example ds = 10 for ten store analysis
def main(df, stores_list):
    run_app(df=df, stores_list=stores_list, save_mapes=True, plot_figs=True, ds=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the script with all_data_ts as an argument")
    parser.add_argument("input_file", help="Path to the input CSV file (all_data_ts)")
    args = parser.parse_args()
    all_data_ts = pd.read_csv(args.input_file, sep=',')
    stores_list = all_data_ts['unique_id'].unique().tolist()
    main(df=all_data_ts, stores_list=stores_list)
