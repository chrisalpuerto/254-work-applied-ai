#AUTHOR: Chris Alpuerto

import numpy as np
import torch as t
from typeguard import typechecked

@typechecked
def dot_product(a: list[float], b: list[float]) -> float:
    """
    Calculates the dot product of two lists of floats.

    Args:
        a (list[float]): The first list of float values.
        b (list[float]): The second list of float values.

    Returns:
        float: The dot product of the two lists.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: You can use standard for-loops or list comprehension if you want to be fancy.
    If you want a challenge, try writing this in 1 line.
    """
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total

@typechecked
def find_largest_dot_product_py(X_data: list[list[float]], Y_data: list[float]) -> int:
    """
    Finds the index of the row in X_data that has the largest dot product with Y_data.
    This is the pure Python implementation.

    Args:
        X_data (list[list[float]]): A list of lists representing the input features.
        Y_data (list[float]): A list representing the target values.

    Returns:
        int: The index of the row in X_data with the largest dot product.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: I recommend using the dot_product function you wrote above to complete this.
    The implementation for this function is pretty straightforward.
    """
    best_index = 0
    best_value = None
    for c in range(len(X_data[0])):
        column = [row[c] for row in X_data]
        value = dot_product(column, Y_data)
        if best_value is None or value > best_value:
            best_value = value
            best_index = c
    return best_index

@typechecked
def dot_product_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the dot product of two NumPy arrays.

    Args:
        a (np.ndarray): The first NumPy array.
        b (np.ndarray): The second NumPy array.

    Returns:
        float: The dot product of the two arrays.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return float(np.dot(a, b))

@typechecked
def find_largest_dot_product_np(X_data: np.ndarray, Y_data: np.ndarray) -> int:
    """
    Finds the index of the row in X_data that has the largest dot product with Y_data.
    This is the NumPy implementation.

    Args:
        X_data (np.ndarray): A NumPy array representing the input features.
        Y_data (np.ndarray): A NumPy array representing the target values.

    Returns:
        int: The index of the row in X_data with the largest dot product.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: This might be a little more tricky to do, the solution itself is pretty short (3 lines)
    but finding the right function in numpy might be difficult.
    """
    products = np.dot(Y_data, X_data)
    return int(np.argmax(products))

@typechecked
def mat_mul_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Performs matrix multiplication on two NumPy arrays.

    Args:
        A (np.ndarray): The first NumPy array (matrix).
        B (np.ndarray): The second NumPy array (matrix).

    Returns:
        np.ndarray: The result of the matrix multiplication.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return np.matmul(A, B)

@typechecked
def mat_mul_t(A: t.Tensor, B: t.Tensor) -> t.Tensor:
    """
    Performs matrix multiplication on two PyTorch tensors.

    Args:
        A (t.Tensor): The first PyTorch tensor (matrix).
        B (t.Tensor): The second PyTorch tensor (matrix).

    Returns:
        t.Tensor: The result of the matrix multiplication.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return t.matmul(A, B)

@typechecked
def dot_product_t(a: t.Tensor, b: t.Tensor) -> t.tensor:
    """
    Calculates the dot product of two PyTorch tensors.

    Args:
        a (t.Tensor): The first PyTorch tensor.
        b (t.Tensor): The second PyTorch tensor.

    Returns:
        t.Tensor: The dot product of the two tensors.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return t.dot(a, b)


def build_square_matrix(X_data: np.ndarray, n: int = 4000) -> np.ndarray:
    rows = X_data[:n]
    return np.resize(rows, (n, n))


def device_name() -> str:
    if t.cuda.is_available():
        return "cuda"
    if t.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    import time
    from data_loader import load_dataset_pd, split_xy

    NORM = "GasProperties_norm.csv"
    COLUMNS = ["T", "P", "TC", "SV"]

    df = load_dataset_pd(NORM)
    X_np, Y_np = split_xy(df)
    X_py = X_np.tolist()
    Y_py = Y_np.tolist()

    print("=" * 60)
    print("2(a) DOT PRODUCT USING A PYTHON LOOP")
    print("=" * 60)
    print(f"X_data {X_np.shape}   Y_data {Y_np.shape}")

    start = time.perf_counter()
    for c in range(len(COLUMNS)):
        column = [row[c] for row in X_py]
        value = dot_product(column, Y_py)
        print(f"  {COLUMNS[c]:>3} . Idx = {value:>20,.4f}")
    loop_each = time.perf_counter() - start

    start = time.perf_counter()
    best_py = find_largest_dot_product_py(X_py, Y_py)
    loop_time = time.perf_counter() - start
    print(f"largest: column {best_py} ({COLUMNS[best_py]})")
    print(f"time (all 4 columns): {loop_time:.4f}s")

    print()
    print("=" * 60)
    print("2(b) DOT PRODUCT USING NUMPY")
    print("=" * 60)

    start = time.perf_counter()
    for c in range(len(COLUMNS)):
        value = dot_product_np(X_np[:, c], Y_np)
        print(f"  {COLUMNS[c]:>3} . Idx = {value:>20,.4f}")
    np_each = time.perf_counter() - start

    start = time.perf_counter()
    best_np = find_largest_dot_product_np(X_np, Y_np)
    np_time = time.perf_counter() - start
    print(f"largest: column {best_np} ({COLUMNS[best_np]})")
    print(f"time (all 4 columns): {np_time:.6f}s")

    print()
    print(f"{'':16}{'loop':>14}{'numpy':>14}{'speedup':>12}")
    print(f"{'find largest':16}{loop_time:>13.6f}s{np_time:>13.6f}s{loop_time / np_time:>11.1f}x")
    print(f"{'per column':16}{loop_each:>13.6f}s{np_each:>13.6f}s{loop_each / np_each:>11.1f}x")
    print(f"same answer: {best_py == best_np}")

    print()
    print("=" * 60)
    print("2(c) MATRIX MULTIPLICATION: 32-BIT vs 64-BIT")
    print("=" * 60)

    N = 4000
    square = build_square_matrix(X_np, N)
    print(f"matrix built from X_data: {square.shape}")

    A32 = square.astype(np.float32)
    start = time.perf_counter()
    R32 = mat_mul_np(A32, A32)
    time32 = time.perf_counter() - start
    print(f"float32: {time32:.4f}s   result {R32.shape} {R32.dtype}")

    A64 = square.astype(np.float64)
    start = time.perf_counter()
    R64 = mat_mul_np(A64, A64)
    time64 = time.perf_counter() - start
    print(f"float64: {time64:.4f}s   result {R64.shape} {R64.dtype}")

    flops = 2 * N ** 3
    print(f"float64 / float32 = {time64 / time32:.2f}x")
    print(f"GFLOPS  float32: {flops / time32 / 1e9:.1f}   float64: {flops / time64 / 1e9:.1f}")

    print()
    print("=" * 60)
    print("2(d) GPU ACCELERATION (PyTorch)")
    print("=" * 60)

    dev = device_name()
    print(f"torch {t.__version__}   device available: {dev}")

    C64 = t.from_numpy(A64)
    start = time.perf_counter()
    RC = mat_mul_t(C64, C64)
    cpu_time = time.perf_counter() - start
    print(f"cpu   float64: {cpu_time:.4f}s   result {tuple(RC.shape)} {RC.dtype}")

    if dev != "cpu":
        gpu_dtype = t.float32 if dev == "mps" else t.float64
        G = C64.to(dev, dtype=gpu_dtype)
        mat_mul_t(G, G)
        if dev == "cuda":
            t.cuda.synchronize()
        else:
            t.mps.synchronize()
        start = time.perf_counter()
        RG = mat_mul_t(G, G)
        if dev == "cuda":
            t.cuda.synchronize()
        else:
            t.mps.synchronize()
        gpu_time = time.perf_counter() - start
        print(f"{dev:<5} {str(gpu_dtype).replace('torch.', ''):8}: {gpu_time:.4f}s   result {tuple(RG.shape)} {RG.dtype}")
        print(f"speedup vs cpu: {cpu_time / gpu_time:.1f}x")
        print(f"GFLOPS on {dev}: {flops / gpu_time / 1e9:.1f}")
        if dev == "mps":
            print("note: MPS does not support float64, so the GPU run above is float32")
    else:
        print("no GPU backend found - run this section on Nautilus for 2(d)")

    print()
    print("=" * 60)
    print("dot_product_t")
    print("=" * 60)
    a = t.tensor(X_np[:, 0])
    b = t.tensor(Y_np)
    print(f"  T . Idx = {dot_product_t(a, b).item():,.4f}")
