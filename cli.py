import argparse
from pathlib import Path
from typing import Annotated, Callable, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    # DirectoryPath,
    FilePath,
    PositiveInt,
    NonNegativeInt,
    AfterValidator,
    ValidationError,
    StringConstraints
)

Strategy = Literal["vanilla", "rscv", "optuna"]
DEFAULT_STRATEGY: Strategy = "vanilla"
DEFAULT_RUNS: int = 50
DEFAULT_SEED: int = 42
DEFAULT_OUTPUT_DIR: Path = Path("results")
DEFAULT_SEP: str = ","
SAVE_EVERY: int = 100

_SEP_ALIASES = {"comma": ",", "semicolon": ";", "tab": "\t", "space": " ", "pipe": "|"}

def _suffix(suffix: str) -> Callable[[Path], Path]:
    def check(path: Path) -> Path:
        if path.suffix.lower() != suffix:
            raise ValueError(f"must be a {suffix} file")
        return path
    return check

def _dir_or_missing(path: Path) -> Path:
    if path.exists() and not path.is_dir():
        raise ValueError("exists but is not a directory")
    return path

def _sep(value: str) -> str:
    return _SEP_ALIASES.get(value.lower(), value)


CsvFile = Annotated[FilePath, AfterValidator(_suffix(".csv"))]    # Must exist and have a .csv extension
JsonFile = Annotated[FilePath, AfterValidator(_suffix(".json"))]  # Must exist and have a .json extension
OutputDir = Annotated[Path, AfterValidator(_dir_or_missing)]      # May not exist yet
CsvSeparator = Annotated[str, StringConstraints(min_length=1, max_length=1)]

class Args(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,            # Strictly enforce type validation without automatic type coercion.
    )
    strategy: Strategy
    runs: PositiveInt       # int > 0
    seed: NonNegativeInt    # int >= 0
    config: JsonFile
    train: CsvFile
    test: CsvFile
    sep: CsvSeparator
    output_dir: OutputDir
    save_every: NonNegativeInt  # int >= 0; 0 = save only at the end

    def prepare_output_dir(self) -> Path:
        """Create the output directory (and parents) if missing. Call it from main.py."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU-based XGBRegressor optimization",
        # epilog="Example: main.py -s optuna -r 100 -t 42",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False
    )

    parser.add_argument(
        "-s", "--strategy",
        default=DEFAULT_STRATEGY,
        type=str,
        choices=get_args(Strategy),
        help="Strategy to use for the runs."
    )
    parser.add_argument(
        "-r", "--runs",
        default=DEFAULT_RUNS,
        type=int,
        help="Number of runs, a positive integer."
    )
    parser.add_argument(
        "-t", "--seed",
        default=DEFAULT_SEED,
        type=int,
        help="Initial seed, a non-negative integer."
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        metavar="JSON",
        help="JSON file with the configuration.",
    )
    parser.add_argument(
        "--train", "--tr",
        type=Path,
        required=True,
        metavar="CSV",
        help="CSV file with the training set.",
    )
    parser.add_argument(
         "--test", "--te",
        type=Path,
        required=True,
        metavar="CSV",
        help="CSV file with the test set.",
    )
    parser.add_argument(
        "--sep",
        type=_sep,
        default=DEFAULT_SEP,
        metavar="CHAR",
        help="Column separator of the CSV files: a single character or one of "
             "comma, semicolon, tab, space, pipe.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Directory where the results are saved (created if missing).",
    )
    parser.add_argument(
        "-e", "--save-every",
        default=SAVE_EVERY,
        type=int,
        metavar="N",
        help="Save intermediate results every N runs; 0 saves only at the end.",
    )
    return parser

def parse_args(argv: list[str] | None = None) -> Args:
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        return Args(**vars(ns))
    except ValidationError as e:
        parser.error("; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'args'}: {err['msg']}"
            for err in e.errors()
        )
    )