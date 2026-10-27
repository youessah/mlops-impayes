import os
import shutil

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from src.config import HF_REPO, RAW_PATH


def find_parquet_file() -> str:
    """Repère le fichier .parquet dans le dépôt Hugging Face."""
    files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
    parquets = [f for f in files if f.endswith(".parquet")]
    if not parquets:
        raise FileNotFoundError(f"Aucun fichier .parquet trouvé. Fichiers : {files}")
    preferred = [f for f in parquets if "telecom_credit_scoring_data" in f]
    return (preferred or parquets)[0]


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    remote_file = find_parquet_file()
    print(f"Fichier distant : {remote_file}")

    local_file = hf_hub_download(repo_id=HF_REPO, filename=remote_file, repo_type="dataset")
    shutil.copy(local_file, RAW_PATH)

    df = pd.read_parquet(RAW_PATH)
    print(f"Données enregistrées : {RAW_PATH}")
    print(f"Lignes : {len(df)} | Colonnes : {df.shape[1]}")
    assert df.shape == (500_000, 15), f"Forme inattendue : {df.shape}"
    print("Contrôle de forme : OK")


if __name__ == "__main__":
    main()