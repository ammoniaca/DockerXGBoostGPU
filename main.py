import logging
from time import perf_counter

import optuna

import cupy as cp
import numpy as np
import numpy.typing as npt

from contextlib import closing

import shap

from xgboost import XGBRegressor
from sklearn.model_selection import KFold

#-------------------------------------------------------------------------------
from cli import Args, parse_args
from src.strategies.vanilla_strategy.vanilla_strategy import VanillaStrategy
from src.strategies.factory_strategy import make_search_strategy
from src.utils.config_loader import load_config
from src.utils.prepare_gpu_data import load_dataset_to_gpu
from src.utils.analysis_metrics import compute_metrics
from src.utils.save_table import save_table
#--------------------------------------------------------------------------

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args: Args = parse_args(argv)

    # Read config and prepare output first: fail fast, before touching the GPU.
    config = load_config(args.config)

    output_dir = args.prepare_output_dir()
    tag = f"{args.train.stem}_{args.strategy}"

    paths = {
        "metrics": output_dir / f"{tag}_metrics.csv",
        "importance": output_dir / f"{tag}_importance.csv",
        "shap": output_dir / f"{tag}_shap.csv",
    }

    # create n random integer seeds
    rng = np.random.default_rng(args.seed)
    seeds: npt.NDArray[np.int64] = rng.integers(low=0, high=2**31 - 1, size=args.runs)

    # read data and load to GPU
    X_train_gpu, y_train_gpu, feature_names_train = load_dataset_to_gpu(
        file_name_path=args.train,
        set_sep=args.sep
    )
    X_test_gpu, y_test_gpu, feature_names_test = load_dataset_to_gpu(
        file_name_path=args.test,
        set_sep=args.sep
    )

    if list(feature_names_train) != list(feature_names_test):
        only_train = sorted(set(feature_names_train) - set(feature_names_test))
        only_test = sorted(set(feature_names_test) - set(feature_names_train))
        raise ValueError(
            "Train and test features differ "
            f"(only in train: {only_train}; only in test: {only_test}) "
            "or have a different order."
        )

    # Same validation splits for all runs. Indices only: no need for the GPU array..
    cv_splits = list(
        KFold(
            n_splits=config.setting.strategy.n_k_folds,
            shuffle=True,
            random_state=args.seed
        )
        .split(X_train_gpu)
    )

    X_test_cpu = cp.asnumpy(X_test_gpu)
    y_test_cpu = cp.asnumpy(y_test_gpu).ravel()

    metrics_rows: list[dict] = []
    shap_rows: list[dict] = []
    importance_rows: list[dict] = []

    start_time_all_runs = perf_counter()
    try:
        for idx, seed in enumerate(seeds, start=1):
            start_time_single_run = perf_counter()
            seed = int(seed)

            logger.info(
                "Run %d/%d | strategy=%s | seed=%d",
                idx,
                len(seeds),
                args.strategy,
                seed
            )

            model_params: dict = {
                **config.setting.model.fixed,
                "random_state": seed,
            }

            if args.strategy == "vanilla":
                model_params.pop("early_stopping_rounds", None)
                model_params["subsample"] = config.setting.strategy.vanilla.subsample
                model_params["colsample_bytree"] = config.setting.strategy.vanilla.colsample_bytree
                strategy = VanillaStrategy(estimator=XGBRegressor(**model_params))
            else:
                strategy = make_search_strategy(
                    name=args.strategy,
                    estimator=XGBRegressor(**model_params),
                    n_trials=config.setting.strategy.n_trials,
                    metric=config.setting.strategy.metric,
                    seed=seed,
                    direction=config.setting.strategy.optuna.direction,
                    sampler=config.setting.strategy.optuna.sampler,
                    pruner=config.setting.strategy.optuna.pruner,
                )

            # close() also frees the GPU memory (BaseStrategy._free_gpu_memory),
            # so optimize() sits inside the with: it is released even if it fails.
            with closing(strategy):
                if args.strategy != "vanilla":
                    strategy.optimize(
                        X=X_train_gpu,
                        y=y_train_gpu,
                        search_space=config.setting.model.suggested,
                        cv_splits=cv_splits
                    )

                fitted_model = strategy.fit(X_train_gpu, y_train_gpu, device="cuda")

                # Model feature importance
                importance = fitted_model.feature_importances_

                # Metrics on the test set
                y_pred_cpu = cp.asnumpy(cp.asarray(fitted_model.predict(X_test_gpu)))
                metrics = compute_metrics(y_true=y_test_cpu, y_pred=y_pred_cpu)

                # TreeSHAP
                explainer = shap.TreeExplainer(fitted_model)
                shap_values = explainer.shap_values(X_test_cpu)
                # shap_df = pd.DataFrame(shap_values, columns=X_test_df.columns)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)  # Global feature importance

                # Append the three rows together, only when all of them are ready:
                # the tables always stay aligned, even if a run fails halfway.
                importance_rows.append(
                    {"run": idx, "seed": int(seed), **dict(zip(feature_names_test, importance))}
                )
                metrics_rows.append(
                    {"run": idx, "seed": seed, **metrics}
                )
                shap_rows.append(
                    {"run": idx, "seed": int(seed), **dict(zip(feature_names_test, mean_abs_shap))}
                )

                del (
                    fitted_model,
                    explainer,
                    shap_values,
                    mean_abs_shap,
                    importance,
                    y_pred_cpu,
                )

            logger.info(
                "Completed run %d/%d | seed=%d | elapsed=%.2f s",
                idx,
                len(seeds),
                seed,
                perf_counter() - start_time_single_run,
            )

            # save_every == 0 -> no intermediate checkpoints (final save only)
            if args.save_every and idx % args.save_every == 0:
                save_table(metrics_rows, paths["metrics"])
                save_table(importance_rows, paths["importance"])
                save_table(shap_rows, paths["shap"])

                logger.info("Results saved after run %d/%d", idx, len(seeds))

    finally:
        save_table(metrics_rows, paths["metrics"])
        save_table(importance_rows, paths["importance"])
        save_table(shap_rows, paths["shap"])

    logger.info(
        "All %d runs completed successfully | elapsed=%.2f s",
        len(seeds),
        perf_counter() - start_time_all_runs
    )
    return 0



if __name__ == '__main__':
    raise SystemExit(main())

# Alternativa su GPU, molto più veloce
# XGBoost calcola i valori SHAP nativamente, e su GPU è molto più rapido del TreeExplainer su CPU.
# Con 1000 run può fare una differenza grande:
# booster = fitted_model.get_booster()
# contribs = booster.predict(xgb.DMatrix(X_test_gpu), pred_contribs=True)
# shap_values = contribs[:, :-1]    # l'ultima colonna è il bias (expected value)
# base_value = contribs[0, -1]