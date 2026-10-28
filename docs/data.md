# Données, prétraitements et feature engineering

## Source
Jeu de données Hugging Face `electricsheepafrica/africa-synth-telecom-telecom-credit-scoring-data-nigeria`
(500 000 lignes, 15 colonnes, **synthétique**). Version suivie par DVC : md5
`517425048d17a19e74ea7d8eb3492de7` (tag Git `data-v1.0`).

## Constats de l'exploration
- Aucune valeur manquante, aucune ligne entièrement dupliquée.
- 13 661 `customer_id` répétés : collisions d'identifiants aléatoires, pas des clients dupliqués.
- `assessment_date` vaut toujours 2025-09-30 : colonne inutile, aucune dimension temporelle.
- Cible `default_risk` à trois classes : low 54,6 %, medium 23,2 %, high 22,3 %.

## Cible
Classification binaire : `1` si `default_risk == "high"`, sinon `0` (22,3 % de positifs).

## Variables conservées
Numériques : `tenure_months`, `avg_monthly_spend_ngn`, `recharge_frequency_per_month`,
`payment_consistency_score`, `late_payment_count`. Catégorielles : `operator`, `account_type`.

## Variables exclues
| Variable | Raison |
|---|---|
| `customer_id`, `phone_number` | identifiants (minimisation des données) |
| `assessment_date` | constante |
| `credit_score`, `credit_rating`, `loan_eligibility`, `max_loan_amount_ngn` | sorties d'un scoring déjà calculé : entrées non indépendantes |

Un essai de contrôle avec ces colonnes (`xgboost_WITH_leakage_cols`, marqué `do_not_deploy`) donne
le même ROC-AUC (0,748) : elles sont redondantes, leur exclusion ne coûte rien.

## Prétraitements (dans le pipeline scikit-learn, sauvegardés avec le modèle)
- `StandardScaler` sur les variables numériques.
- `OneHotEncoder(handle_unknown="ignore")` sur les variables catégorielles.
- Découpage 80 / 20 stratifié, `random_state = 42`.
- Déséquilibre : `class_weight="balanced"` (régression logistique), `scale_pos_weight` (XGBoost).

## Contrôle de qualité
`src/validate.py` : schéma Pandera (types, plages, modalités autorisées). Un échec arrête le pipeline.