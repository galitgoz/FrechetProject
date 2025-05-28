import pandas as pd
import re
import os
from typing import List, Tuple, Dict, Callable

Curve = List[Tuple[float, float]]
_parsers: Dict[str, Callable[..., pd.DataFrame]] = {}

def register_parser(name: str):
    """
    Decorator to register a parser function for a dataset type.
    """
    def decorator(fn: Callable[..., pd.DataFrame]):
        _parsers[name] = fn
        return fn
    return decorator

def parse_dataset(name: str, *args, **kwargs) -> pd.DataFrame:
    """
    Dispatch function to parse a dataset using the registered parser.
    """
    if name not in _parsers:
        raise KeyError(f"No parser registered under '{name}'")
    return _parsers[name](*args, **kwargs)

@register_parser('hurdat2')
def parse_hurdat2(filepath: str) -> pd.DataFrame:
    """
    Parse HURDAT2 storm data file to a standard DataFrame.
    """
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    storm_id, storm_name = None, None
    for line in lines:
        parts = line.strip().split(',')
        if re.match(r'^[A-Z]{2}\d{6}$', parts[0].strip()):
            storm_id = parts[0].strip()
            storm_name = parts[1].strip()
            continue
        else:
            if storm_id is None:
                continue
            datetime = parts[0].strip() + parts[1].strip()
            lat = float(parts[4][:-1]) * (1 if parts[4][-1] == 'N' else -1)
            lon = float(parts[5][:-1]) * (-1 if parts[5][-1] == 'W' else 1)
            data.append({
                'id': storm_id,
                'name': storm_name,
                'datetime': pd.to_datetime(datetime, format='%Y%m%d%H%M', errors='coerce'),
                'lat': lat,
                'lon': lon
            })
    df = pd.DataFrame(data)
    df.dropna(subset=['datetime'], inplace=True)
    return df[['id', 'datetime', 'lat', 'lon']]

@register_parser('movebank_cat')
def parse_movebank_cat(filepath: str) -> pd.DataFrame:
    """
    Parse Movebank cat CSV to standard DataFrame.
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df.rename(columns={
        'individual-local-identifier': 'id',
        'timestamp': 'datetime',
        'location-lat': 'lat',
        'location-long': 'lon'
    }, inplace=True)
    return df[['id', 'datetime', 'lat', 'lon']]

@register_parser('taxi_data')
def parse_taxi_file(filepath: str) -> pd.DataFrame:
    """
    Parse taxi log .txt file into DataFrame with columns: id, datetime, lat, lon
    """
    df = pd.read_csv(filepath, header=None, names=['id', 'datetime', 'lon', 'lat'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    return df[['id', 'datetime', 'lat', 'lon']]

@register_parser('gagliardo_pigeon')
def parse_gagliardo_pigeon(filepath: str) -> pd.DataFrame:
    """
    Parse Gagliardo et al. (2016) pigeon navigation data to standard DataFrame.
    Uses columns: individual-local-identifier (id), timestamp (datetime), location-lat (lat), location-long (lon).
    """
    df = pd.read_csv(filepath)
    df.rename(columns={
        'individual-local-identifier': 'id',
        'timestamp': 'datetime',
        'location-lat': 'lat',
        'location-long': 'lon'
    }, inplace=True)
    # Parse datetime with seconds and optional milliseconds
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    # Fallback for rows without milliseconds
    mask_na = df['datetime'].isna()
    if mask_na.any():
        df.loc[mask_na, 'datetime'] = pd.to_datetime(df.loc[mask_na, 'datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df[['id', 'datetime', 'lat', 'lon']]
    df.dropna(subset=['datetime', 'lat', 'lon'], inplace=True)
    return df

