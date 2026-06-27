from datetime import datetime

import pandas as pd


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(pd.UTC).replace(tzinfo=None)
    return dt

def resample_timeseries(df: pd.DataFrame, freq: str = '1T') -> pd.DataFrame:
    df = df.set_index('time')
    df = df.resample(freq).mean().interpolate(method='linear')
    return df.reset_index()
