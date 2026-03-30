# Fiche de révision — Classification partielle multi‑objectif

## Contexte
Notebook principal : [Multi_PartialClassif_v2 (2).ipynb](Multi_PartialClassif_v2%20(2).ipynb)
Objectif : optimiser simultanément Confiance (Precision) et Sensibilité (Recall) via NSGA‑II / SPEA2 en représentation Michigan (une règle par solution).

---

## Notions clés
- Pareto / non‑dominance : une solution A domine B si A est au moins aussi bonne sur tous les objectifs et strictement meilleure sur au moins un. Le front Pareto = solutions non‑dominées.
- Hypervolume (HV) : volume en espace objectif entre le front et un point de référence (`ref_point`). Plus grand = meilleur front (dans l'espace minimisation).
- `ref_point` : vecteur pire que le pire observé pour chaque objectif (ex. `worst + 5%`).
- JMetalPy minimise : pour maximiser une métrique M, on passe `-M` comme objectif.
- Représentation Michigan : solution = vecteur `2 * n_attributes` → paires `(lower_i, upper_i)` ; attribut actif si `lower_i <= upper_i`.
- Protocole de sélection : sélectionner runs par HV sur TRAIN, choisir solution par meilleur F1 sur TRAIN, évaluer sur TEST.

---

## Sens des métriques (rappel + formules)
- True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN)

- Precision (Confiance)
  - Formule : Precision = TP / (TP + FP)
  - Sens : parmi les prédictions positives, proportion de vrais positifs. Utile quand FP coûte cher.

- Recall (Sensibilité)
  - Formule : Recall = TP / (TP + FN)
  - Sens : parmi les vrais positifs, proportion détectée. Utile quand FN coûte cher.

- F1‑score
  - Formule : F1 = 2 * (Precision * Recall) / (Precision + Recall)
  - Sens : moyenne harmonique entre precision et recall, utile pour équilibre.

- Accuracy (Exactitude)
  - Formule : Accuracy = (TP + TN) / (TP + TN + FP + FN)
  - Sens : proportion de prédictions correctes ; trompeuse si classes déséquilibrées.

- Hypervolume (HV)
  - Interprétation : volume dominé par le front jusqu'au `ref_point`. En minimisation, on veut front proche de l'origine négative (si on minimise -precision, -recall). Choisir `ref_point` légèrement pire que la pire valeur observée.

---

## Protocole expérimental (pas‑à‑pas)
1. Préparation des données
   - Split stratifié train/test 70/30 (seed fixe) — déjà dans le notebook.
2. Baselines (Scikit‑Learn)
   - RF, SVM, DecisionTree (C4.5-like) — 3‑fold CV pour precision/recall/f1 ± std.
3. Étude de convergence (choix MAX_EVALUATIONS)
   - Fixer `pop_size = 50`, budgets = `[100,250,500,1000,2500,5000,7500,10000]`.
   - Pour chaque budget : 1 run (seed fixe), calculer HV (TRAIN), tracer HV vs budget (log scale).
   - Détecter plateau : gain marginal < 1% → choisir `MAX_EVALUATIONS` (ou valeur légèrement supérieure si tu veux de la marge).
4. Expérimentations principales
   - Paramètres typiques : `POP_SIZES = [20,50,100]`, `N_RUNS = 20` (ou 5–10 pour tests rapides), `MAX_EVALUATIONS` choisi par convergence.
   - Pour chaque (algo, pop_size) lancer `N_RUNS` runs (seeds différents), collecter fronts non‑dominés et HV (TRAIN).
5. Sélection
   - Pour chaque algo/dataset : choisir run avec HV maximal (TRAIN).
   - Sur ce front, choisir la solution avec meilleur F1 sur TRAIN (solution interprétable).
   - Évaluer cette solution sur TEST : precision/recall/f1/accuracy.
6. Analyse & reporting
   - Boxplots HV par pop_size, scatter Precision–Recall (TRAIN & TEST), décoder règle et commenter plausibilité métier.
   - Rapporter mean ± std du HV et temps de calcul.

---

## Checklist pratique (à suivre dans le TP)
- Relire : `PartialClassifMO`, `compute_hv`, `run_convergence_study`, `run_mo_algorithm`.
- Test rapide :
```python
POP_SIZES = [20]
MAX_EVALUATIONS = 200
N_RUNS = 3
res_quick = run_experiments('NSGA-II', X_pima_tr, y_pima_tr)
```
- Ajouter un paramètre (ex. facteur mutation) :
```python
# en haut, paramètres
MUTATION_PROB_FACTOR = 1.2

# dans run_mo_algorithm
n_vars = 2 * X_train.shape[1]
mutation_prob = MUTATION_PROB_FACTOR * (1.0 / n_vars)
```
- Passer à 3 objectifs (Precision, Recall, Accuracy) :
  - Mettre `number_of_objectives()` à 3, ajouter `solution.objectives[2] = -accuracy_score(...)`, et fixer `REF_POINT` de longueur 3 (ex. `[0.05,0.05,0.05]`). Utiliser `compute_hv_3obj` si besoin.

---

## Snippets utiles
- HV 3 objets (copier si tu ajoutes Accuracy) :
```python
from jmetal.core.quality_indicator import HyperVolume
import numpy as np

def compute_hv_3obj(solutions, ref_point=None):
    if not solutions:
        return 0.0
    front = np.array([s.objectives for s in solutions])  # shape (n,3)
    if ref_point is None:
        worst = front.max(axis=0)
        margin = 0.05 * np.maximum(1.0, np.abs(worst))
        ref_point = list(worst + margin)
    hv_calc = HyperVolume(ref_point)
    return hv_calc.compute(front)
```

- Run rapide de vérification :
```python
POP_SIZES = [20]
MAX_EVALUATIONS = 200
N_RUNS = 3
res_quick = run_experiments('NSGA-II', X_pima_tr, y_pima_tr)
```

---

## Interprétation rapide des sorties
- HV élevé + faible dispersion → configuration fiable.
- Boxplot : comparer médianes et dispersion par pop_size.
- Scatter Precision–Recall : points vers haut‑droite = meilleurs compromis.
- Règle décodée : indiquer attributs actifs et intervalles, commenter plausibilité.
- Généralisation : si F1_test ≈ F1_train → bon ; si F1_test ≪ F1_train → surapprentissage.

---

## Phrases prêtes pour la présentation
- "Nous avons optimisé Confiance et Sensibilité via AG MO (NSGA‑II / SPEA2)."
- "Le budget `MAX_EVALUATIONS` a été choisi par étude de convergence (plateau HV → X évaluations)."
- "La taille de population optimale observée est X (trade‑off HV / coût)."
- "La règle finale (2 attributs) : SI A ∈ [a,b] ET B ∈ [c,d] → positif ; F1_train = X, F1_test = Y."
- "Les modèles Scikit‑Learn ont souvent F1 supérieur mais ne donnent pas le front de compromis explicite."

---

## Rapide plan de rapport (3–5 slides)
1. Contexte & objectif
2. Méthode & protocole
3. Résultats clés (HV, front, règle, metrics test)
4. Discussion & limites
5. Conclusion & perspectives

---

Si tu veux une version condensée (1 page) ou que j'insère automatiquement un lien/entrée dans le notebook, dis "1 page" ou "insère".
