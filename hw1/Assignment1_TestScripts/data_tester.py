#AUTHOR: Chris Alpuerto

import numpy as np
import pandas as pd
import time
from pathlib import Path

from data_loader import (
    load_dataset,
    load_dataset_np,
    normalize_array,
    normalize_array_np,
    load_dataset_pd,
    split_xy,
    split_training_test,
)

HERE = Path(__file__).parent # so the script runs from any cwd
RAW = str(HERE / "data" / "GasProperties.csv")
NORM = str(HERE / "GasProperties_norm.csv")


if __name__ == "__main__":
    '''
    You can write your filled out functions in here to test them out.
    '''
    start = time.perf_counter()
    GasProperties = load_dataset(RAW)
    py_rows = normalize_array(GasProperties, NORM)
    py_time = time.perf_counter() - start

    start = time.perf_counter()
    GasProperties_np = load_dataset_np(RAW)
    np_rows = normalize_array_np(GasProperties_np)
    np_time = time.perf_counter() - start

    print(f"python rows: {py_rows}")
    print(f"numpy  rows: {np_rows}")
    assert py_rows == np_rows, "the two implementations disagree on the row count"

    print(f"python time: {py_time:.3f}s")
    print(f"numpy  time: {np_time:.3f}s")
    print(f"speedup: {py_time / np_time:.1f}x")

    df = load_dataset_pd(NORM)
    X_data, Y_data = split_xy(df)
    X_train, Y_train, X_test, Y_test = split_training_test(X_data, Y_data)

    print(f"X_train {X_train.shape}  Y_train {Y_train.shape}")
    print(f"X_test  {X_test.shape}  Y_test  {Y_test.shape}")
