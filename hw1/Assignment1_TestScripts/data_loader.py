'''
This file is where you migrate all the functions that you made in data_process1 and data_process2.
'''

#AUTHOR: Chris Alpuerto
import numpy as np
import pandas as pd
import csv
import time
from typeguard import typechecked


@typechecked
def load_dataset(filename: str) -> list[list[float]]:
    """
    Loads a dataset from a CSV file into a list of lists of floats.
    The first row (header) is skipped.
    """
    try:
        dataset = []
        with open(filename, "r") as file:
            reader = csv.reader(file) # Reader object
            header = next(reader)  # Skip the header row
            for row in reader:
                dataset.append([float(x) for x in row]) # each row -> array of float values
        return dataset
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []


@typechecked
def load_dataset_np(filename: str) -> np.ndarray:
    """
    Loads a dataset from a CSV file into a NumPy array.
    The first row (header) is skipped.
    """
    try:
        data = np.genfromtxt(filename, delimiter=",", skip_header=1)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return np.array([]) # typeguard requires an ndarray, not a list


@typechecked
def normalize_array(arr: list[list[float]], out_file: str | None = None) -> int:
    """
    Normalizes the input array (list of lists) using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.
    """
    if not arr:
        return 0

    n = len(arr)
    n_feat = len(arr[0]) - 1 # last column is idx, left alone

    mean, minimum, maximum, std = [], [], [], []
    for c in range(n_feat):
        col = [row[c] for row in arr]
        mu = sum(col) / n
        var = sum((x - mu) ** 2 for x in col) / n
        mean.append(mu)
        minimum.append(min(col))
        maximum.append(max(col))
        std.append(var ** 0.5)

    normalized = []
    for row in arr:
        if any(abs(row[c] - mean[c]) > 2 * std[c] for c in range(n_feat)): # outlier
            continue
        new_row = [(row[c] - mean[c]) / (maximum[c] - minimum[c]) for c in range(n_feat)]
        new_row.append(row[-1]) # idx carried over unchanged
        normalized.append(new_row)

    if out_file is not None:
        with open(out_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["T", "P", "TC", "SV", "Idx"])
            writer.writerows(normalized)

    return len(normalized)


@typechecked
def normalize_array_np(arr: np.ndarray, out_file: str | None = None) -> int:
    """
    Normalizes the input NumPy array using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.
    """
    if arr.size == 0:
        return 0

    features = arr[:, :-1] # every column except idx
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    minimum = features.min(axis=0)
    maximum = features.max(axis=0)

    keep = np.all(np.abs(features - mean) <= 2 * std, axis=1) # drops the outlier rows
    normalized = (features[keep] - mean) / (maximum - minimum)
    normalized = np.column_stack((normalized, arr[keep, -1]))

    if out_file is not None:
        np.savetxt(out_file, normalized, delimiter=",", header="T,P,TC,SV,Idx", comments="")

    return int(normalized.shape[0]) # cast so typeguard sees a real int


@typechecked
def load_dataset_pd(filename: str) -> pd.DataFrame:
    """
    Loads normalized data from a CSV file into a Pandas DataFrame.
    """
    return pd.read_csv(filename)


@typechecked
def split_xy(df: pd.DataFrame, y_axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a Pandas DataFrame into feature (X) and target (Y) NumPy arrays.
    """
    Y_data = df.iloc[:, y_axis].to_numpy()
    X_data = df.drop(df.columns[y_axis], axis=1).to_numpy() # everything but the y column
    return X_data, Y_data


@typechecked
def split_training_test(
        X_data: np.ndarray,
        Y_data: np.ndarray,
        split: float = 0.8
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the input feature (X) and target (Y) NumPy arrays into training and testing sets.
    """
    cut = int(len(X_data) * split) # 80/20 by default
    return X_data[:cut], Y_data[:cut], X_data[cut:], Y_data[cut:]


if __name__ == "__main__":
    RAW = "data/GasProperties.csv"
    NORM = "GasProperties_norm.csv"

    print("--- load_dataset ---")
    start = time.perf_counter()
    GasProperties = load_dataset(RAW)
    load_py = time.perf_counter() - start
    print(f"rows: {len(GasProperties)}  columns: {len(GasProperties[0])}")
    print(f"first row: {GasProperties[0]}")
    print(f"time: {load_py:.3f}s")

    print("\n--- load_dataset_np ---")
    start = time.perf_counter()
    GasProperties_np = load_dataset_np(RAW)
    load_np = time.perf_counter() - start
    print(f"shape: {GasProperties_np.shape}")
    print(f"first row: {GasProperties_np[0]}")
    print(f"time: {load_np:.3f}s")

    print("\n--- normalize_array ---")
    start = time.perf_counter()
    py_rows = normalize_array(GasProperties, NORM)
    norm_py = time.perf_counter() - start
    print(f"rows kept: {py_rows} (dropped {len(GasProperties) - py_rows} outliers)")
    print(f"wrote: {NORM}")
    print(f"time: {norm_py:.3f}s")

    print("\n--- normalize_array_np ---")
    start = time.perf_counter()
    np_rows = normalize_array_np(GasProperties_np)
    norm_np = time.perf_counter() - start
    print(f"rows kept: {np_rows} (dropped {len(GasProperties_np) - np_rows} outliers)")
    print(f"time: {norm_np:.3f}s")
    print(f"matches python result: {py_rows == np_rows}")

    print("\n--- performance comparison ---")
    print(f"{'':12}{'python':>10}{'numpy':>10}{'speedup':>10}")
    print(f"{'load':12}{load_py:>9.3f}s{load_np:>9.3f}s{load_py / load_np:>9.1f}x")
    print(f"{'normalize':12}{norm_py:>9.3f}s{norm_np:>9.3f}s{norm_py / norm_np:>9.1f}x")

    print("\n--- load_dataset_pd ---")
    df = load_dataset_pd(NORM)
    print(f"shape: {df.shape}  columns: {list(df.columns)}")
    print(df.head())

    print("\n--- split_xy ---")
    X_data, Y_data = split_xy(df)
    print(f"X_data {X_data.shape} (T, P, TC, SV)")
    print(f"Y_data {Y_data.shape} (Idx)")

    print("\n--- split_training_test ---")
    X_train, Y_train, X_test, Y_test = split_training_test(X_data, Y_data)
    print(f"X_train {X_train.shape}  Y_train {Y_train.shape}")
    print(f"X_test  {X_test.shape}  Y_test  {Y_test.shape}")
