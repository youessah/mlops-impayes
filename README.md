# MLOps - Prédiction des impayés télécom

Industrialisation d'un modèle de prédiction du risque d'impayé (`default_risk = high`) sur le jeu
de données *Telecom Credit Scoring Data (Nigeria)* : 500 000 lignes, 15 colonnes (données synthétiques).

## Architecture

Données (DVC) -> validation (Pandera) -> entraînement (scikit-learn / XGBoost, suivi MLflow)
-> registre de modèles -> image Docker + API FastAPI -> monitoring (PSI, Evidently) -> réentraînement.

| Composant | Outil |
|---|---|
| Versionnement du code | Git / GitHub |
| Versionnement des données | DVC (stockage distant local) |
| Qualité des données | Pandera (`src/validate.py`) |
| Suivi des expériences et registre | MLflow (SQLite local) |
| Serving | FastAPI + Uvicorn |
| Conteneurisation | Docker (image CPU, `xgboost-cpu`) |
| CI/CD/CT | GitHub Actions (`.github/workflows/mlops.yml`) |
| Monitoring | PSI/KS (`src/monitor.py`) + Evidently (environnement isolé) |

## Démarrage (Windows, PowerShell)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m src.data_ingestion      # télécharge le dataset (ou: dvc pull)
    python -m src.validate            # contrat de qualité des données
    python -m src.train               # entraînement, MLflow, porte de validation
    python -m pytest -q               # tests de l'API
    docker build -t impayes-api:1.0.0 .
    docker compose up -d              # API sur http://127.0.0.1:8000/docs

## Monitoring

    python -m src.simulate_traffic --n 500 --seed 7            # trafic normal
    python -m src.simulate_traffic --n 500 --seed 8 --drift    # trafic avec dérive simulée
    python -m src.monitor                                      # décision + reports/drift_report.html

Le rapport Evidently s'exécute dans un second environnement (`.venv-evidently`, pandas < 3) car
Evidently 0.7.23 est incompatible avec pandas 3.1 :

    python -m venv .venv-evidently
    .\.venv-evidently\Scripts\python.exe -m pip install "pandas<3" evidently

## Résultats

Voir `docs/model_card.md` et `docs/data.md`. Modèle retenu : XGBoost, F1 = 0,563 sur le jeu de test.