#AUTHOR: Chris Alpuerto

import numpy as np
import pandas as pd
import csv
import time # Import the time module
from typeguard import typechecked

@typechecked
def load_dataset(filename: str) -> list[list[float]]:
    """
    Loads a dataset from a CSV file into a list of lists of floats.
    The first row (header) is skipped.

    Args:
        filename (str): The path to the CSV file. 

    Returns:
        list[list[float]]: A list of lists, where each inner list represents a row
                           of the dataset with float values.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: If you are stuck on this, I recommend looking through the python csv library.
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

    Args:
        filename (str): The path to the CSV file.

    Returns:
        np.ndarray: A NumPy array representing the dataset.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Numpy has a very useful csv file reader called genfromtxt.
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

    Args:
        arr (list[list[float]]): The input dataset as a list of lists of floats.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Probably the most complicated function to write because you cant use numpy.
    I would spend some time on this to make sure that all the equations for metrics are correct.
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

    Args:
        arr (np.ndarray): The input dataset as a NumPy array.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: The same function as normalize_array but using numpy to calculate the metrics.
    This function should be almost copy and paste with numpy functions.
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


if __name__ == "__main__":
    RAW = "data/GasProperties.csv"
    NORM = "GasProperties_norm.csv"

    print("--- load_dataset (python) ---")
    start = time.perf_counter()
    GasProperties = load_dataset(RAW)
    load_py = time.perf_counter() - start
    print(f"rows loaded: {len(GasProperties)}")
    print(f"columns: {len(GasProperties[0])}")
    print(f"first row: {GasProperties[0]}")
    print(f"time: {load_py:.3f}s")

    print("\n--- load_dataset_np (numpy) ---")
    start = time.perf_counter()
    GasProperties_np = load_dataset_np(RAW)
    load_np = time.perf_counter() - start
    print(f"shape: {GasProperties_np.shape}")
    print(f"first row: {GasProperties_np[0]}")
    print(f"time: {load_np:.3f}s")

    print("\n--- normalize_array (python) ---")
    start = time.perf_counter()
    py_rows = normalize_array(GasProperties, NORM)
    norm_py = time.perf_counter() - start
    print(f"rows kept: {py_rows} of {len(GasProperties)}")
    print(f"rows dropped as outliers: {len(GasProperties) - py_rows}")
    print(f"wrote: {NORM}")
    print(f"time: {norm_py:.3f}s")

    print("\n--- normalize_array_np (numpy) ---")
    start = time.perf_counter()
    np_rows = normalize_array_np(GasProperties_np)
    norm_np = time.perf_counter() - start
    print(f"rows kept: {np_rows} of {len(GasProperties_np)}")
    print(f"rows dropped as outliers: {len(GasProperties_np) - np_rows}")
    print(f"time: {norm_np:.3f}s")

    print("\n--- performance comparison ---")
    print(f"{'':12}{'python':>10}{'numpy':>10}{'speedup':>10}")
    print(f"{'load':12}{load_py:>9.3f}s{load_np:>9.3f}s{load_py / load_np:>9.1f}x")
    print(f"{'normalize':12}{norm_py:>9.3f}s{norm_np:>9.3f}s{norm_py / norm_np:>9.1f}x")
    print(f"{'total':12}{load_py + norm_py:>9.3f}s{load_np + norm_np:>9.3f}s"
          f"{(load_py + norm_py) / (load_np + norm_np):>9.1f}x")
    print(f"\nboth row counts match: {py_rows == np_rows}")

