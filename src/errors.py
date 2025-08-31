from typing import Sequence, Literal, overload

import pandas as pd
import streamlit as st
from pandas import DataFrame

class NoFileSubmitted(Exception):
    def __init__(self):
        message = 'Please upload a file.'
        super().__init__(message)

class FileMissingColumn(Exception):
    def __init__(self, columns:Sequence[str]):
        message = f'Input file is missing the following columns: {', '.join(columns)}. Maybe check the column names (case sensitive)? Please fix it and try again'
        super().__init__(message)

class NoData(Exception):
    def __init__(self):
        message = 'Input contains only header row and no data. Please provide data and try again.'
        super().__init__(message)

class MissingValue(Exception):
    def __init__(self):
        super().__init__()

class NaNColumn(Exception):
    def __init__(self, columns: Sequence[str]):
        message = f'The following columns contain only missing values: {", ".join(columns)}. Please fix it and try again'
        super().__init__(message)

def check_missing_cols(df:DataFrame, ref_df:DataFrame):
    if not set(ref_df.columns).issubset(set(df.columns)):
        raise FileMissingColumn(set(ref_df.columns)-set(df.columns))

def check_na(df: DataFrame):
    missing_counts = df.isna().sum()
    nrows = df.shape[0]
    if (nrows > 0) and (missing_counts == nrows).any():
        cols = missing_counts[missing_counts == df.shape[0]].index.tolist()
        raise NaNColumn(cols)
    if missing_counts.any():
        raise MissingValue()
    
def quick_fix_na_val(df_or_buffer, option: Literal['impute with mean', 'impute with median', 'drop']) -> DataFrame:
    if hasattr(df_or_buffer, 'read'):
        df_or_buffer.seek(0)            # Tip: Any file handler needs to be reset for repeated reads
        df = pd.read_csv(df_or_buffer)
    else:
        df = df_or_buffer
        
    if option == 'impute with mean':
        return df.fillna(df.mean(skipna=True))
    elif option == 'impute with median':
        return df.fillna(df.median(skipna=True))
    elif option == 'drop':
        return df.dropna()
    else:
        raise ValueError(f"Unknown option: {option}")
    
def check_type(df: DataFrame, ref_df: DataFrame):           # Improvement: compare non-numeric types (but since we're not planning on using other datasets, this will do)
    mismatched = []
    for col in ref_df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')       # LEARN
        if df[col].isnull().any():
            mismatched.append(col)
    if mismatched:
        raise TypeError(f"Non-numeric values found in '{', '.join(mismatched)}'. Please fix and try again")

def validate_input(df_or_buffer, ref_df: pd.DataFrame):
    warnings = []
    if df_or_buffer is None:
        raise NoFileSubmitted()
    if hasattr(df_or_buffer, 'read'):
        df = pd.read_csv(df_or_buffer)
        check_missing_cols(df, ref_df)
        if set(df.columns) != set(ref_df.columns):
            warnings.append(f'Input containing extra column(s). They\'ll be ignored.')
            df = df[ref_df.columns]
    else:
        df = df_or_buffer
    if df.shape[0] == 0:
        raise NoData()
    check_na(df)
    check_type(df, ref_df)
    return df, warnings