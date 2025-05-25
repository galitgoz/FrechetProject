import os
import glob
import pandas as pd
import sys
import time
from src.frechet.io import parse_dataset

def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    percent = (iteration / float(total)) * 100
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')
    if iteration == total:
        print()

def export_to_standard_csv(
    dataset_type: str,
    input_path: str,
    output_path: str
):
    """
    Parse a dataset and export to a standardized CSV with columns: id, date, hour, lat, lon.
    """
    try:
        df = parse_dataset(dataset_type, input_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse '{input_path}': {e}")
        return
    if 'datetime' not in df.columns:
        print(f"[ERROR] 'datetime' column missing in {input_path}")
        return

    df['date'] = df['datetime'].dt.strftime('%Y%m%d')
    df['hour'] = df['datetime'].dt.strftime('%H%M')
    df_std = df[['id', 'date', 'hour', 'lat', 'lon']].copy()
    df_std.to_csv(output_path, index=False)
    print(f"[OK] Exported {len(df_std)} rows to '{output_path}'.")

def export_taxi_directory(input_dir: str, output_path: str):
    """
    Export all taxi data files from a directory into a single standardized CSV.
    """
    all_files = glob.glob(os.path.join(input_dir, "*.txt"))
    all_dfs = []
    total = len(all_files)
    for idx, filename in enumerate(all_files, 1):
        try:
            df = parse_dataset('taxi_data', filename)
        except Exception as e:
            print(f"[ERROR] Failed to parse '{filename}': {e}")
            continue
        for col in ['datetime', 'id', 'lat', 'lon']:
            if col not in df.columns:
                print(f"[ERROR] '{col}' column missing in {filename}")
                break
        else:
            df['date'] = df['datetime'].dt.strftime('%Y%m%d')
            df['hour'] = df['datetime'].dt.strftime('%H%M')
            df = df[['id', 'date', 'hour', 'lat', 'lon']]
            all_dfs.append(df)
        print_progress_bar(idx, total, prefix='Processing taxi files:', suffix='Complete', length=40)
    if not all_dfs:
        print(f"[WARNING] No valid taxi files found in {input_dir}")
        return
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"[OK] Exported {len(combined_df)} rows to '{output_path}'.")

def combine_csvs(input_paths, output_path):
    """
    Combine multiple CSVs into a single CSV.
    """
    dfs = []
    for path in input_paths:
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            continue
        dfs.append(pd.read_csv(path))
    if not dfs:
        print("[ERROR] No files to combine.")
        return
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"[OK] Combined {len(dfs)} files into '{output_path}'.")

def main():
    # Storms
    export_to_standard_csv(
        dataset_type='hurdat2',
        input_path='../data/hurdat2-1851-2024-040425.txt',
        output_path='../data/storms.csv'
    )

    # Taxi
    export_taxi_directory(
        input_dir='../data/taxi_log_2008_by_id/',
        output_path='../data/taxi.csv'
    )

    # Cats
    export_to_standard_csv(
        dataset_type='movebank_cat',
        input_path='../data/Pet Cats United States.csv',
        output_path='../data/cats.csv'
    )

    # Combine storms & taxi
    combine_csvs(['../data/storms.csv', '../data/taxi.csv'], '../data/all.csv')
    print("All data exported successfully.")

if __name__ == '__main__':
    main()
