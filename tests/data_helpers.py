from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qtyche_qrc.data.config import DataPreparationConfig, load_data_config
from qtyche_qrc.data.features import FEATURE_NAMES
from qtyche_qrc.data.fixtures import create_or_verify_fixture_snapshots
from qtyche_qrc.data.pipeline import prepare_data
from qtyche_qrc.models.dataset import ModelDataset, load_model_dataset


def canonical_frame(rows: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    index = np.arange(rows, dtype=float)
    spy_close = 100.0 * np.exp(np.cumsum(0.001 + 0.004 * np.sin(index / 4.0)))
    qqq_close = 80.0 * np.exp(np.cumsum(0.0012 + 0.003 * np.cos(index / 5.0)))
    spy_open = spy_close * (1.0 + 0.001 * np.sin(index))
    return pd.DataFrame(
        {
            "date": dates,
            "spy_open": spy_open,
            "spy_high": np.maximum(spy_open, spy_close) * 1.01,
            "spy_low": np.minimum(spy_open, spy_close) * 0.99,
            "spy_close": spy_close,
            "spy_adjusted_close": spy_close,
            "spy_volume": 1_000_000 + index * 2_500 + (index % 7) * 1_000,
            "vix_close": 15.0 + 2.0 * np.sin(index / 6.0),
            "qqq_close": qqq_close,
            "qqq_volume": 800_000 + index * 1_700,
        }
    )


def write_test_data_config(root: Path) -> DataPreparationConfig:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    split_config = {
        "schema_version": 1,
        "split": {
            "strategy": "chronological",
            "shuffle": False,
            "require_forward_window_containment": True,
            "purge_trading_days": 5,
            "boundaries": {
                "train": {"start": "2020-01-01", "end": "2020-06-30"},
                "validation": {"start": "2020-07-01", "end": "2020-09-30"},
                "test": {"start": "2020-10-01", "end": "2020-12-31"},
            },
        },
    }
    data_config = {
        "schema_version": 1,
        "experiment": {
            "name": "test_data",
            "seed": 7,
            "output_dir": "data/processed",
        },
        "data": {
            "mode": "cached_csv",
            "project_root": "..",
            "raw_paths": {
                "spy": "data/raw/fixture_spy.csv",
                "vix": "data/raw/fixture_vix.csv",
                "qqq": "data/raw/fixture_qqq.csv",
            },
            "processed_path": "data/processed",
            "symbols": {"spy": "SPY", "vix": "^VIX", "qqq": "QQQ"},
            "source_date_range": {"start": "2019-10-01", "end": "2021-01-15"},
            "observation_date_range": {"start": "2020-01-01", "end": "2020-12-31"},
            "required_columns": [
                "date",
                "spy_open",
                "spy_high",
                "spy_low",
                "spy_close",
                "spy_adjusted_close",
                "spy_volume",
                "vix_close",
                "qqq_close",
                "qqq_volume",
            ],
            "features": list(FEATURE_NAMES),
            "targets": {
                "definition_version": "qtyche_volatility_regime_v1",
                "primary_classification": "target_regime_5d",
                "primary_transition": "target_transition",
                "secondary_regression": "target_rv_5d",
                "horizon_trading_days": 5,
                "annualization_factor": 252,
                "regime_quantiles": [0.33, 0.66],
            },
            "split_config": "splits.yaml",
            "missing_data_policy": "drop_secondary_and_report",
        },
    }
    (config_dir / "splits.yaml").write_text(
        yaml.safe_dump(split_config, sort_keys=False), encoding="utf-8"
    )
    data_path = config_dir / "data.yaml"
    data_path.write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")
    return load_data_config(data_path)


def write_test_public_data_config(root: Path, *, mode: str = "cached_csv") -> DataPreparationConfig:
    """Write a public-market variant with isolated raw and processed locations."""

    fixture = write_test_data_config(root)
    root_value = yaml.safe_load(fixture.source.read_text(encoding="utf-8"))
    data = root_value["data"]
    snapshot_id = "test_public_market_v1"
    snapshot_dir = f"data/raw/public_market/{snapshot_id}"
    data.update(
        {
            "mode": mode,
            "data_source_type": "public_market",
            "is_synthetic": False,
            "snapshot": {
                "id": snapshot_id,
                "provider": "yahoo_chart",
                "manifest": f"{snapshot_dir}/snapshot_manifest.json",
            },
            "raw_paths": {
                "spy": f"{snapshot_dir}/spy.csv",
                "vix": f"{snapshot_dir}/vix.csv",
                "qqq": f"{snapshot_dir}/qqq.csv",
            },
            "processed_path": "data/processed/public_market",
        }
    )
    root_value["experiment"]["output_dir"] = "data/processed/public_market"
    path = root / "configs/data_public_market.yaml"
    path.write_text(yaml.safe_dump(root_value, sort_keys=False), encoding="utf-8")
    return load_data_config(path)


def prepared_model_dataset(root: Path) -> ModelDataset:
    config = write_test_data_config(root)
    create_or_verify_fixture_snapshots(config)
    prepare_data(config)
    return load_model_dataset(config.processed_path)


def write_test_model_config(
    root: Path,
    model_type: str = "majority_classifier",
    task: str = "regime_classification",
) -> Path:
    config_dir = root / "configs" / "models"
    config_dir.mkdir(parents=True, exist_ok=True)
    selection_metric = "macro_f1" if task == "regime_classification" else "qlike"
    config = {
        "schema_version": 1,
        "experiment": {
            "name": "test_baseline",
            "project_root": "../..",
            "seed": 19,
            "output_root": "results/test",
        },
        "data": {
            "processed_dir": "data/processed",
            "manifest": "data/processed/data_manifest.json",
        },
        "model": {"type": model_type, "task": task, "parameters": {}},
        "search": {
            "enabled": False,
            "selection_metric": selection_metric,
            "maximum_trials": 1,
            "space": {},
        },
        "evaluation": {"transition_threshold": 0.5, "variance_floor": 1e-12},
    }
    path = config_dir / "model.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
