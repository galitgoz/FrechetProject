import os
import pandas as pd
import glob
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from .io import parse_dataset


def make_storms_csv(input_path: str, output_path: str):
    storms_df = parse_dataset('hurdat2', input_path)
    storms_df['date'] = storms_df['datetime'].dt.strftime('%Y%m%d')
    storms_df['hour'] = storms_df['datetime'].dt.strftime('%H%M')
    storms_df[['id', 'date', 'hour', 'lan', 'lon']].to_csv(output_path, index=False)

def make_taxi_csv(input_dir: str, output_path: str):
    all_files = glob.glob(os.path.join(input_dir, "*.txt"))
    all_dfs = []

    for filename in all_files:
        df = parse_dataset('taxi_data', filename)
        df['date'] = df['datetime'].dt.strftime('%Y%m%d')
        df['hour'] = df['datetime'].dt.strftime('%H%M')
        df = df[['id', 'date', 'hour', 'lan', 'lon']]
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)

def make_cats_csv(input_path: str, output_path: str):
    cats_df = parse_dataset('movebank_cat', input_path)
    cats_df['date'] = cats_df['timestamp'].dt.strftime('%Y%m%d')
    cats_df['hour'] = cats_df['timestamp'].dt.strftime('%H%M')
    cats_df.rename(columns={'location-lat': 'lan', 'location-long': 'lon', 'individual-local-identifier': 'id'}, inplace=True)
    cats_df[['id', 'date', 'hour', 'lan', 'lon']].to_csv(output_path, index=False)

def make_combined_csv(storm_csv: str, taxi_csv: str, combined_csv: str):
    storms_df = pd.read_csv(storm_csv)
    taxi_df = pd.read_csv(taxi_csv)
    combined_df = pd.concat([storms_df, taxi_df], ignore_index=True)
    combined_df.to_csv(combined_csv, index=False)


def main():
    make_storms_csv('../data/hurdat2-1851-2024-040425.txt', '../data/storms.csv')
    make_taxi_csv('../data/taxi_log_2008_by_id/', '../data/taxi.csv')
    make_cats_csv('../data/Pet Cats United States.csv', '../data/cats.csv')
    make_combined_csv('../data/storms.csv', '../data/taxi.csv', '../data/all.csv')

    print("All data exported successfully.")


if __name__ == '__main__':
    main()
