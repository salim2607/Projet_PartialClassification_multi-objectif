# Classification Partielle Multi-Objectif avec JMetalPy

**M2 MIAGE - Optimisation et Aide a la Decision (OAFD) | Universite de Lille**

---

## Presentation du projet

Ce projet implemente un systeme de **classification partielle multi-objectif** utilisant des algorithmes evolutionnaires (NSGA-II et SPEA2) via le framework JMetalPy. L'objectif est de decouvrir automatiquement des **regles de classification interpretables** (de type SI...ALORS) en optimisant simultanement deux metriques antagonistes : la **precision** (Confiance) et le **rappel** (Sensibilite).

Les regles sont comparees aux classifieurs classiques Scikit-Learn (Random Forest, SVM, C4.5) pour evaluer leur competitivite tout en conservant leur avantage en termes d'interpretabilite.

---

## Datasets

| Dataset | Instances | Attributs | Positifs | Negatifs | Tache |
|---|---|---|---|---|---|
| `pima_diabetes.csv` | 768 | 8 | 268 | 500 | Detection du diabete (Glucose, BMI, Age...) |
| `yeast1.csv` | 1484 | 8 | 463 | 1021 | Localisation cellulaire de proteines de levure |

Les deux datasets sont des problemes de **classification binaire desequilibree** : la classe positive est minoritaire, ce qui rend le compromis precision/rappel particulierement pertinent.

---

## Prerequis et Installation

```bash
pip install jmetalpy scikit-learn numpy pandas matplotlib
```

Versions testees : jmetalpy>=1.5.5, scikit-learn>=1.0, numpy>=1.21, pandas>=1.3, matplotlib>=3.4

---

## Structure du projet

```
Projet_PartialClassification_multi-objectif/
├── Multi_PartialClassif_v2.ipynb   # Notebook principal (code commente)
├── pima_diabetes.csv               # Dataset PIMA Indians Diabetes
├── yeast1.csv                      # Dataset Yeast1
├── boxplots_hv.png                 # Boxplots HV par taille de population
├── pareto_fronts_sklearn.png       # Fronts Pareto vs Scikit-Learn (train)
├── comparison_test.png             # Comparaison Precision/Recall (test)
├── convergence_study.png           # Courbe de convergence HV vs budget
└── README.md
```

---

## Description detaillee de chaque cellule du notebook

### Cellule 1 - Imports et dependances

Charge toutes les bibliotheques requises. Doit etre executee en premier.

**Bibliotheques standard** :
- `random` : generateur pseudo-aleatoire (graines pour reproductibilite)
- `warnings` : suppression des alertes non critiques de JMetalPy et sklearn
- `time` : mesure des temps d'execution des algorithmes
- `numpy` : calcul numerique vectorise (tableaux, statistiques, opérations matricielles)
- `pandas` : lecture et manipulation des fichiers CSV en DataFrames
- `matplotlib` : trace des graphiques (boxplots, scatter plots, courbes de convergence)

**Scikit-Learn** :
- `StratifiedKFold` : validation croisee preservant les proportions de classes positives
- `train_test_split` : decoupage stratifie train/test (70%/30%)
- `RandomForestClassifier` : foret aleatoire (boite noire, 100 arbres)
- `SVC` : Support Vector Machine (boite noire, noyau RBF)
- `DecisionTreeClassifier` : arbre de decision C4.5-like (boite blanche, max_depth=4)
- `precision_score`, `recall_score`, `f1_score`, `accuracy_score` : metriques d'evaluation

**JMetalPy** :
- `Problem`, `FloatSolution` : classes de base pour un probleme MO a variables reelles
- `PolynomialMutation` : mutation polynomiale controlee par `distribution_index` (grand = petite perturbation)
- `SBXCrossover` : Simulated Binary Crossover, recombine deux parents dans l'espace reel
- `StoppingByEvaluations` : critere d'arret base sur le nombre d'evaluations
- `get_non_dominated_solutions` : filtre les solutions non-dominees (front de Pareto)
- `NSGAII` : Non-dominated Sorting Genetic Algorithm II (Deb et al. 2002)
- `SPEA2` : Strength Pareto Evolutionary Algorithm 2 (Zitzler et al. 2001)
- `HyperVolume` : calcul du volume de l'espace objectif domine par le front

---

### Cellule 2 - Chargement des donnees

Lit les deux fichiers CSV et produit les matrices X (features) et vecteurs y (labels).

```python
X_pima  = df_pima.drop('Outcome', axis=1).to_numpy()  # Matrice 768x8
y_pima  = df_pima['Outcome'].to_numpy()               # Labels 0=sain, 1=diabetique

X_yeast = df_yeast.drop('Output', axis=1).to_numpy()  # Matrice 1484x8
y_yeast = df_yeast['Output'].to_numpy()               # Labels 0=autre, 1=cytoplasme
```

Affiche les dimensions, le nombre de positifs/negatifs, et les noms d'attributs de chaque dataset.

---

### Cellule 3 - Baselines Scikit-Learn (validation croisee 3-fold)

Evalue trois classifieurs en **validation croisee stratifiee a 3 plis** pour obtenir des references.

**Fonction `run_sklearn(X, y, model_fn, n_folds=3)`** :
1. Cree 3 plis equilibres avec `StratifiedKFold`
2. Pour chaque pli : entraine le modele sur 2/3 des donnees, predit sur 1/3
3. Calcule precision, recall, F1 sur le pli de test
4. Retourne les moyennes et ecarts-types sur les 3 plis

| Classifieur | Type | Parametres |
|---|---|---|
| Random Forest (RF) | Boite noire | n_estimators=100, random_state=42 |
| SVM | Boite noire | kernel='rbf', random_state=42 |
| C4.5 (DecisionTree) | Boite blanche | max_depth=4, random_state=42 |

Resultats stockes dans `sklearn_results[dataset][algo]` pour la comparaison finale.

---

### Cellule 4 - Classe PartialClassifMO et fonctions utilitaires

**Coeur du projet.** Definit le probleme d'optimisation multi-objectif via la classe `Problem` de JMetalPy.

#### Representation Michigan (une regle par individu)

Chaque individu encode une unique regle de classification :

```
SI attr_0 dans [lo_0, hi_0] ET attr_1 dans [lo_1, hi_1] ET ... --> classe positive
```

Pour un dataset a n attributs, chaque solution contient 2n variables reelles :
- `variables[2i]`   = borne inferieure `lo_i` de l'attribut i
- `variables[2i+1]` = borne superieure `hi_i` de l'attribut i

Un attribut est active si `lo_i <= hi_i`. Si `lo_i > hi_i`, l'attribut est ignore.

#### Objectifs (minimisation dans JMetalPy)

| Objectif | Formule stockee | Signification reelle |
|---|---|---|
| `objectives[0]` | `-Precision = -TP/(TP+FP)` | Qualite des predictions positives |
| `objectives[1]` | `-Recall = -TP/(TP+FN)` | Taux de detection des positifs reels |

JMetalPy minimise : minimiser -f est equivalent a maximiser f.

#### Methodes de la classe PartialClassifMO

**`__init__(X, y)`** : stocke les donnees, identifie les indices positifs/negatifs, calcule les bornes [min_attribut, max_attribut+1] pour l'espace de recherche (l'extension +1 permet des intervalles desactives avec lo > hi).

**`_decode(solution)`** : traduit le vecteur de 2n variables en predictions binaires. Pour chaque exemple j, predit 1 si TOUS les attributs actives ont leur valeur dans l'intervalle correspondant. Retourne zeros si aucun attribut n'est active.

**`evaluate(solution)`** : appelle `_decode`, puis calcule `-precision_score` et `-recall_score`. Si y_pred.sum()==0 (regle vide), les deux objectifs valent 0.0.

**`create_solution()`** : genere un individu initial valide en selectionnant 1 exemple positif + 1 negatif, activant 2 attributs aleatoires avec un intervalle [min(vp,vn), max(vp,vn)], desactivant les autres (lo = vp+1 > hi = vp).

#### Fonctions utilitaires

**`predict(solution_vars, X)`** : applique la regle encodee sur un dataset X quelconque (train ou test).

**`decode_rule(solution_vars, feature_names)`** : affiche la regle en langage naturel.
Exemple de sortie : `Regle : SI Glucose dans [120.500, 180.200] ET BMI dans [30.100, 45.000] --> classe positive`

**`compute_hv(solutions, ref_point=[0.05, 0.05])`** : calcule l'hypervolume du front.
Construit la matrice des objectifs puis appelle `HyperVolume(ref_point).compute(front)`.

**`get_pareto_front(solutions)`** : retourne uniquement les solutions non-dominees via `get_non_dominated_solutions`.

---

### Cellule 5 - Parametres experimentaux et split train/test

| Parametre | Valeur | Description |
|---|---|---|
| `CROSSOVER_PROB` | 0.9 | Probabilite d'application du croisement SBX |
| `CROSSOVER_INDEX` | 20.0 | Grand indice = enfants proches des parents (exploitation) |
| `MUTATION_INDEX` | 20.0 | Grand indice = petites perturbations (stabilite) |
| `MAX_EVALUATIONS` | 500 | Budget d'evaluations (defini apres etude de convergence) |
| `N_RUNS` | 20 | Runs independants par configuration (robustesse statistique) |
| `REF_POINT` | [0.05, 0.05] | Point de reference HV |
| `POP_SIZES` | [20, 50, 100] | 3 tailles de population comparees |

Split 70%/30% stratifie (random_state=42) :
- PIMA : 537 train / 231 test
- Yeast1 : 1038 train / 446 test

---

### Cellule 6 - Etude de convergence

Calibrage empirique du budget `MAX_EVALUATIONS` optimal.

Teste 8 budgets [100, 250, 500, 1000, 2500, 5000, 7500, 10000] avec NSGA-II sur PIMA (representatif).
Pour chaque budget : 1 run -> calcul HV -> trace de la courbe.

**`find_plateau(budgets, hvs, threshold=0.01)`** : detecte le premier budget ou le gain relatif `(HV[i]-HV[i-1])/HV[i-1]` passe sous 1%.

Sortie : `convergence_study.png` (echelle log en X) avec ligne verticale rouge sur le budget retenu.

---

### Cellule 7 - Fixation definitive de MAX_EVALUATIONS

Cellule interactive : l'utilisateur lit la courbe et modifie manuellement `MAX_EVALUATIONS`.
Valeur retenue : **500 evaluations**.

---

### Cellule 8 - Fonctions d'execution des algorithmes

**`run_mo_algorithm(algo_name, X_train, y_train, pop_size, max_evaluations, seed)`**

Lance un run unique :
1. Fixe `random.seed(seed)` et `np.random.seed(seed)`
2. Instancie `PartialClassifMO(X_train, y_train)`
3. Configure `PolynomialMutation(prob=1/n_vars)` et `SBXCrossover(prob=0.9)`
4. Cree `StoppingByEvaluations(max_evaluations)`
5. Execute `algo.run()` et retourne `algo.get_result()` (population finale)

**`run_experiments(algo_name, X_train, y_train, pop_sizes, n_runs)`**

Lance 20 runs independants par taille de population (seed = 100*run + 42).
Pour chaque run : extrait le front de Pareto, calcule l'HV.
Retourne `{pop_size: {"hv": [20 valeurs], "fronts": [20 fronts]}}`.

---

### Cellules 9 a 12 - Execution des experiences principales

| Cellule | Algorithme | Dataset | Variable |
|---|---|---|---|
| 9 | NSGA-II | PIMA | `nsgaii_pima` |
| 10 | NSGA-II | Yeast1 | `nsgaii_yeast` |
| 11 | SPEA2 | PIMA | `spea2_pima` |
| 12 | SPEA2 | Yeast1 | `spea2_yeast` |

**Total : 240 runs** (2 algos x 2 datasets x 3 pop_sizes x 20 runs).

---

### Cellule 13 - Boxplots de l'hypervolume

Grille 2x2 de boxplots (NSGA-II/SPEA2 x PIMA/Yeast1) comparant la distribution HV sur 20 runs.

**Lecture** : mediane (trait noir), boite Q1-Q3, moustaches, outliers.
**Interpretation** : boite haute et etroite = algorithme performant ET stable.

Sortie : `boxplots_hv.png`

---

### Cellule 14 - Selection du meilleur front Pareto

**`get_best_run`** : identifie parmi les 60 runs le front avec l'HV maximal.

**`front_to_prec_rec`** : convertit `objectives[i]` negatifs en Precision/Recall positifs.

**`best_f1_solution`** : selectionne dans le front la solution maximisant le F1 sur X_train.
Cette solution sert de representant unique pour la comparaison avec Scikit-Learn.

Structure du dict `best[dataset][algo]` :
- `pop` : taille de population du meilleur run
- `front` : liste des solutions non-dominees du meilleur run
- `hv` : hypervolume maximal atteint
- `best_sol` : solution avec le meilleur F1 sur le train
- `best_f1_train` : valeur du F1 correspondant

---

### Cellule 15 - Visualisation des fronts Pareto (train)

Nuage de points (Recall en X, Precision en Y) avec :
- Points bleus : front NSGA-II
- Points oranges : front SPEA2
- Etoile noire : solution F1-max de chaque algorithme
- Symboles colores : baselines RF/SVM/C4.5

Sortie : `pareto_fronts_sklearn.png`

---

### Cellule 16 - Evaluation sur le jeu de test

Evalue la meilleure solution de chaque front sur le TEST pour mesurer la generalisation.
Calcule Precision, Recall, F1, Accuracy sur train et test.
Affiche la regle decouverte sous forme lisible via `decode_rule`.

---

### Cellule 17 - Tableau recapitulatif comparatif

DataFrame pandas comparant RF, SVM, C4.5, NSGA-II, SPEA2 sur le TEST :
Precision, Recall, F1, HV, Type (boite noire ou blanche).

---

### Cellule 18 - Graphe comparatif Precision/Recall (test)

Identique a la cellule 15 mais avec toutes les evaluations sur le TEST.
Detecte un eventuel surapprentissage des regles MO.

Sortie : `comparison_test.png`

---

### Cellule 19 - Graphes de convergence de l'hypervolume

Courbes HV moyen +/- 1 ecart-type en fonction des evaluations pour NSGA-II et SPEA2.
La bande coloree semi-transparente represente la variabilite inter-runs.

Sortie : `convergence_hv.png`

---

## Indicateur de qualite : l'Hypervolume

L'**Hypervolume (HV)** est la mesure principale de comparaison des fronts de Pareto.

**Definition** : surface de l'espace objectif 2D dominee par le front, delimitee par le point de reference.

```
HV = surface entre le front de Pareto et le point de reference [0.05, 0.05]
```

**Proprietes** :
- Capture a la fois la **convergence** (front proche de l'origine) et la **diversite** (front etale)
- Point de reference [0.05, 0.05] correspond a Precision > 0.95 ET Recall > 0.95 comme ideal
- Plus HV est eleve, meilleur est le front (convergence + diversite combinees)

---

## Metriques d'evaluation

| Metrique | Formule | Role dans le projet |
|---|---|---|
| Precision (Confiance) | TP / (TP + FP) | Objectif 1 : negation stockee (-Precision) |
| Rappel (Sensibilite) | TP / (TP + FN) | Objectif 2 : negation stockee (-Recall) |
| F1-mesure | 2 x P x R / (P + R) | Selection de la meilleure solution du front |
| Accuracy | (TP + TN) / Total | Information complementaire sur le test |
| Hypervolume (HV) | Volume domine | Comparaison globale des fronts de Pareto |

---

## Protocole experimental

```
1. Etude de convergence
   NSGA-II x PIMA x 8 budgets --> MAX_EVALUATIONS = 500

2. Experiences principales (240 runs)
   2 algos x 2 datasets x 3 pop_sizes x 20 runs
   Seeds : 42, 142, 242, ..., 1942

3. Analyse
   Boxplots HV --> taille de population optimale
   Meilleur front (HV max) --> visualisation Pareto
   Meilleure solution F1 --> comparaison avec Scikit-Learn
   Evaluation sur le test --> generalisation
   Regles decouvertes --> interpretabilite
```

---

## Ordre d'execution recommande

1. Cellule 1 : Imports
2. Cellule 2 : Chargement des donnees
3. Cellule 3 : Baselines Scikit-Learn
4. Cellule 4 : Classe PartialClassifMO et utilitaires
5. Cellule 5 : Parametres et split train/test
6. Cellule 6 : Etude de convergence
7. Cellule 7 : Fixer MAX_EVALUATIONS
8. Cellule 8 : Fonctions d'execution
9. Cellules 9-12 : 240 runs (NSGA-II et SPEA2 sur les 2 datasets)
10. Cellule 13 : Boxplots HV
11. Cellule 14 : Selection des meilleurs fronts
12. Cellule 15 : Graphes Pareto (train)
13. Cellule 16 : Evaluation test + regles lisibles
14. Cellule 17 : Tableau comparatif final
15. Cellule 18 : Graphe comparatif (test)
16. Cellule 19 : Graphes de convergence

---

## Fichiers generes

| Fichier | Description |
|---|---|
| `boxplots_hv.png` | Grille 2x2 : HV par taille de population (NSGA-II et SPEA2) |
| `pareto_fronts_sklearn.png` | Fronts Pareto + baselines Scikit-Learn (espace Precision/Recall, train) |
| `comparison_test.png` | Comparaison Precision/Recall sur le jeu de test |
| `convergence_study.png` | Courbe HV = f(budget) pour le calibrage de MAX_EVALUATIONS |
| `convergence_hv.png` | Evolution HV moyen +/- std au fil des evaluations |

---

## Avantages de l'approche multi-objectif

| Aspect | Classifieurs Scikit-Learn | Algorithmes MO (ce projet) |
|---|---|---|
| Resultat | Un seul modele (un point) | Un front de N compromis possibles |
| Interpretabilite | Limitee (RF/SVM boites noires) | Maximale : regle SI...ALORS lisible |
| Flexibilite | Fixe apres entrainement | Le decideur choisit son compromis P/R |
| Indicateur de qualite | F1 seule | Hypervolume (convergence + diversite) |

---

*Projet M2 MIAGE - OAFD - Universite de Lille*
