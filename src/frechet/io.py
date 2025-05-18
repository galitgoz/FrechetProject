import pandas as pd
import re
from typing import List, Tuple, Dict, Callable

_parsers: Dict[str, Callable[..., pd.DataFrame]] = {}

def register_parser(name: str):
    def decorator(fn: Callable[..., pd.DataFrame]):
        _parsers[name] = fn
        return fn
    return decorator

def parse_dataset(name: str, *args, **kwargs) -> pd.DataFrame:
    if name not in _parsers:
        raise KeyError(f"No parser registered under '{name}'")
    return _parsers[name](*args, **kwargs)

@register_parser('hurdat2')
def parse_hurdat2(filepath: str) -> pd.DataFrame:
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    storm_id, storm_name = None, None
    for line in lines:
        parts = line.strip().split(',')
        if re.match(r'^[A-Z]{2}\d{6}$', parts[0].strip()):
            storm_id = parts[0].strip()
            storm_name = parts[1].strip()
            continue  # header line, move to next line for data entries
        else:
            if storm_id is None:
                continue  # safety check
            datetime = parts[0].strip() + parts[1].strip()
            lat = float(parts[4][:-1]) * (1 if parts[4][-1] == 'N' else -1)
            lon = float(parts[5][:-1]) * (-1 if parts[5][-1] == 'W' else 1)
            data.append({
                'id': storm_id,
                'name': storm_name,
                'datetime': pd.to_datetime(datetime, format='%Y%m%d%H%M', errors='coerce'),
                'lan': lat,
                'lon': lon
            })

    df = pd.DataFrame(data)
    df.dropna(subset=['datetime'], inplace=True)
    return df[['id', 'datetime', 'lan', 'lon']]

@register_parser('movebank_cat')
def parse_movebank_cat(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df[['individual-local-identifier', 'timestamp', 'location-lat', 'location-long']]

@register_parser('taxi_data')
def parse_taxi_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None, names=['id', 'datetime', 'lon', 'lan'])
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    return df