from pathlib import Path
import cupy as cp
import pandas as pd

def load_dataset_to_gpu(
        file_name_path: str | Path,
        set_sep: str
) -> tuple[cp.ndarray, cp.ndarray, list[str]]:
    # Read the CSV file into a pandas DataFrame on the CPU
    dataset_cpu = pd.read_csv(file_name_path, sep=set_sep)

    # Extract feature names, excluding the target in the last column.
    feature_names = dataset_cpu.columns[:-1].tolist()

    # Convert the DataFrame to a NumPy array before transferring it to the GPU.
    # All columns must contain numeric data.
    dataset_gpu = cp.asarray(dataset_cpu)

    # Split the features and target, assuming the target is the last column.
    X_gpu = dataset_gpu[:, :-1]
    y_gpu = dataset_gpu[:, -1]

    return X_gpu, y_gpu, feature_names