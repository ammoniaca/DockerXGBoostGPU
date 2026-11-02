from pathlib import Path
import pandas as pd

def save_table(rows: list[dict], path: Path) -> None:
    """Atomically save rows to a CSV file: a crash during writing will not corrupt the previous file."""
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)