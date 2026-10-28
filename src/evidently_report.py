"""Rapport Evidently, exécuté dans l'environnement isolé .venv-evidently (pandas < 3)."""
import sys
from importlib.metadata import version

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


def main() -> None:
    ref_path, cur_path, out_path = sys.argv[1:4]
    ref = pd.read_csv(ref_path)
    cur = pd.read_csv(cur_path)
    numeric = [c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c])]
    categorical = [c for c in ref.columns if c not in numeric]
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)
    result = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(cur, data_definition=definition),
        Dataset.from_pandas(ref, data_definition=definition),
    )
    result.save_html(out_path)
    print(f"Evidently {version('evidently')} (pandas {pd.__version__})")


if __name__ == "__main__":
    main()