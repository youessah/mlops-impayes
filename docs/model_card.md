# Fiche modèle - impayes-default-risk (v1)

## Usage prévu
Score de risque d'impayé pour des clients télécom, en appui à une décision humaine
(relance, plafond de crédit). Ne doit pas servir seul à refuser un service.

## Modèle
XGBoost (300 arbres, profondeur 6), précédé d'un pipeline de prétraitement. Run MLflow
`61a49c32380247c0923a3ed626fe9b94`, version du registre 1, alias `staging`.

## Performances (jeu de test : 100 000 lignes, seuil 0,5)
| Modèle | Précision | Rappel | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Régression logistique (baseline) | 0,456 | 0,691 | 0,550 | 0,745 | 0,436 |
| **XGBoost (retenu)** | 0,480 | 0,681 | **0,563** | 0,748 | 0,438 |
| XGBoost avec colonnes suspectes (non déployable) | 0,496 | 0,697 | 0,579 | 0,748 | 0,441 |

## Limites
- Données synthétiques : les performances ne sont pas transposables à un cas réel.
- Le plafond (ROC-AUC proche de 0,75) vient des données, pas du réglage du modèle.
- Les scores ne sont pas calibrés (pondération des classes) : ce sont des scores de risque.
- Environ une alerte sur deux est un faux positif au seuil 0,5.
- Aucune analyse d'équité par segment n'a été menée.

## Monitoring et réentraînement
Réentraînement déclenché si au moins 2 variables dépassent un PSI de 0,25, ou si le score dérive,
ou si le F1 baisse de 0,08 ou plus, et seulement si la qualité des données est correcte.
Le nouveau modèle passe par la porte de validation (F1 >= 0,52) avant promotion.