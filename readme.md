# 📋 TP5 – Classification Partielle Multi-objectif avec JMetalPy
### M2 MIAGE – OAFD | Université de Lille

> **Rendu :** Rapport PDF + Code Python dans un fichier `.zip`  
> **Travail :** Binôme autorisé  
> **Datasets :** `yeast1` et `pima_diabète`

---

## 📁 Structure du projet recommandée

```
TP5_ClassifPartielle_MO/
│
├── data/
│   ├── yeast1.csv
│   └── pima_diabete.csv
│
├── src/
│   ├── problem_partialclassif_MO.py     # Définition du problème MO
│   ├── nsga2_runner.py                  # Exécution NSGA-II
│   ├── autre_ga_runner.py               # Exécution 2ème algorithme MO
│   ├── sklearn_baselines.py             # RF, SVM, C4.5
│   ├── tuning.py                        # Tuning des hyperparamètres
│   └── analysis.py                      # Boxplots, fronts Pareto, tableaux
│
├── results/
│   ├── pareto_fronts/
│   ├── boxplots/
│   └── tables/
│
├── rapport.pdf
└── README.md
```

---

## ✅ CHECKLIST GLOBALE DU PROJET

### Phase 0 – Préparation
- [ ] Récupérer le code du TP3 (algorithme génétique mono-objectif)
- [ ] Récupérer les résultats TP3 pour C4.5, Random Forest et SVM (precision + recall)
- [ ] Vérifier que JMetalPy est installé (`pip install jmetalpy`)
- [ ] Vérifier que sklearn est installé (`pip install scikit-learn`)
- [ ] Charger et vérifier les deux datasets : `yeast1` et `pima_diabète`

---

## ✅ CHECKLIST PARTIE 1 – Transformation en Multi-objectif

### 1.1 – Modifier la définition du problème

- [ ] Ouvrir le fichier de définition du problème du TP3
- [ ] Changer le nombre d'objectifs de **1** à **2** dans JMetalPy :
  ```python
  self.number_of_objectives = 2
  ```
- [ ] Modifier la fonction `evaluate()` pour retourner **deux objectifs séparés** :
  ```python
  def evaluate(self, solution):
      # Calculer y_pred à partir des règles encodées
      precision = sklearn.metrics.precision_score(y_test, y_pred)
      recall    = sklearn.metrics.recall_score(y_test, y_pred)
      
      # JMetalPy minimise → on inverse le signe pour maximiser
      solution.objectives[0] = -precision   # Maximiser la confiance
      solution.objectives[1] = -recall      # Maximiser la sensibilité
      return solution
  ```
- [ ] Vérifier que la représentation des individus (bornes + activation) reste identique au TP3
- [ ] Tester que le problème s'instancie sans erreur

### 1.2 – Vérifier les opérateurs génétiques

- [ ] Vérifier que les opérateurs de **sélection** sont compatibles multi-objectif
- [ ] Vérifier que les opérateurs de **croisement** fonctionnent toujours
- [ ] Vérifier que les opérateurs de **mutation** fonctionnent toujours

---

## ✅ CHECKLIST PARTIE 2 – Implémentation des 2 Algorithmes MO

### 2.1 – Algorithme 1 : NSGA-II (obligatoire)

- [ ] Importer NSGA-II depuis JMetalPy :
  ```python
  from jmetal.algorithm.multiobjective.nsgaii import NSGAII
  ```
- [ ] Configurer NSGA-II avec les paramètres de base :
  - [ ] Taille de population (à tuner)
  - [ ] Nombre de générations
  - [ ] Opérateurs de croisement et mutation
- [ ] Tester une exécution complète sur `yeast1`
- [ ] Tester une exécution complète sur `pima_diabète`
- [ ] Vérifier que le front de Pareto est bien retourné

### 2.2 – Algorithme 2 : Autre AG MO (au choix)

Choisir **un** des algorithmes suivants dans JMetalPy :
- [ ] **SPEA2** *(recommandé)*
- [ ] MOEA/D
- [ ] SMS-EMOA
- [ ] GDE3

- [ ] Importer l'algorithme choisi depuis JMetalPy
- [ ] Configurer avec les mêmes paramètres de base que NSGA-II (pour comparaison équitable)
- [ ] Tester une exécution complète sur `yeast1`
- [ ] Tester une exécution complète sur `pima_diabète`
- [ ] Vérifier que le front de Pareto est bien retourné

---

## ✅ CHECKLIST PARTIE 3 – Tuning des paramètres

> **Règle :** On ne tune QUE la taille de population (3 valeurs à choisir)

### 3.1 – Définir le protocole de tuning

- [ ] Choisir 3 tailles de population à tester, par exemple :
  - [ ] Petite : `pop_size = 50`
  - [ ] Moyenne : `pop_size = 100`
  - [ ] Grande : `pop_size = 200`
- [ ] Fixer tous les autres paramètres (générations, probabilités de croisement/mutation)
- [ ] Documenter les choix dans un tableau :

| Paramètre | Valeur fixée |
|---|---|
| Nombre de générations | ? |
| Probabilité de croisement | ? |
| Probabilité de mutation | ? |
| Taille de population | 50 / 100 / 200 |

### 3.2 – Exécuter le tuning

- [ ] Lancer **20 runs** par algorithme × par taille de population × par dataset
  - Total minimum : `2 algos × 3 pop_size × 2 datasets × 20 runs = 240 exécutions`
- [ ] Calculer l'**Hypervolume (HV)** pour chaque run
- [ ] Sauvegarder tous les résultats dans des fichiers CSV

### 3.3 – Analyser les résultats du tuning

- [ ] Produire un **boxplot par algorithme** (HV en fonction de la pop_size)
- [ ] Produire un boxplot pour `yeast1` et un pour `pima_diabète`
- [ ] Identifier la meilleure taille de population pour chaque algorithme
- [ ] Rédiger une **interprétation** des boxplots dans le rapport

---

## ✅ CHECKLIST PARTIE 4 – Comparaison des 2 algorithmes MO

- [ ] Utiliser la meilleure configuration trouvée au tuning pour chaque algorithme
- [ ] Lancer 20 runs pour NSGA-II et 20 runs pour l'autre AG (sur les 2 datasets)
- [ ] Calculer l'HV pour chaque run
- [ ] Produire un boxplot NSGA-II vs Autre AG (HV)
- [ ] Réaliser un **test statistique** si possible (ex: Wilcoxon)
- [ ] Rédiger une conclusion sur quel algorithme est le meilleur

---

## ✅ CHECKLIST PARTIE 5 – Comparaison avec Sklearn

### Étape 1 – Visualiser les fronts de Pareto

- [ ] Pour chaque algorithme, récupérer les 20 fronts de Pareto obtenus sur le **training**
- [ ] Calculer l'HV pour chaque front
- [ ] **Sélectionner le run avec le meilleur HV** comme front représentatif
- [ ] Tracer un graphique (axe X = Sensibilité, axe Y = Confiance) pour chaque dataset
  - [ ] Afficher les 20 fronts en gris clair (transparents)
  - [ ] Afficher le meilleur front en couleur vive
  - [ ] **Sauvegarder la Figure 1**

### Étape 2 – Comparaison sur le TRAINING

- [ ] Récupérer les scores **precision** et **recall** du TP3 pour :
  - [ ] C4.5 (sur training, `yeast1`)
  - [ ] Random Forest (sur training, `yeast1`)
  - [ ] SVM (sur training, `yeast1`)
  - [ ] C4.5 (sur training, `pima_diabète`)
  - [ ] Random Forest (sur training, `pima_diabète`)
  - [ ] SVM (sur training, `pima_diabète`)
- [ ] Sur le même graphique que l'Étape 1, ajouter ces 3 points (C4.5, RF, SVM)
- [ ] **Sauvegarder la Figure 2**

### Étape 3 – Évaluation sur le TEST

- [ ] Prendre le meilleur front Pareto (meilleur HV training) pour chaque AG
- [ ] Pour chaque solution du front, **calculer precision et recall sur le test**
- [ ] Calculer la **F1-mesure** sur le test pour chaque solution
- [ ] Choisir la solution avec le **meilleur F1-score** sur le training pour la présenter
- [ ] Évaluer cette solution sur le test
- [ ] **Sauvegarder la Figure 3** (solution choisie + front Pareto sur test)
- [ ] Remplir le tableau de résultats :

| Métrique | RF | SVM | C4.5 | NSGA-II | Autre AG |
|---|---|---|---|---|---|
| Confiance (training) | | | | | |
| Sensibilité (training) | | | | | |
| F1-mesure (training) | | | | | |
| Confiance (test) | | | | | |
| Sensibilité (test) | | | | | |
| F1-mesure (test) | | | | | |
| Hypervolume | / | / | / | | |

> Répéter ce tableau pour `yeast1` ET `pima_diabète`

---

## ✅ CHECKLIST PARTIE 6 – Analyse et Interprétabilité

### 6.1 – Analyse des résultats

- [ ] Comparer les performances de NSGA-II vs Autre AG (HV, F1, Precision, Recall)
- [ ] Comparer les AG MO avec les algorithmes sklearn
- [ ] Discuter du compromis **confiance / sensibilité** observé sur les fronts Pareto
- [ ] Expliquer pourquoi les AG MO produisent un front et pas un seul point
- [ ] Analyser les différences entre `yeast1` et `pima_diabète`

### 6.2 – Analyse de l'interprétabilité

- [ ] Extraire au moins **une règle interprétable** trouvée par NSGA-II sur chaque dataset
- [ ] Comparer avec les règles de C4.5 (aussi interprétable)
- [ ] Comparer avec RF et SVM (boîtes noires)
- [ ] Rédiger un paragraphe sur l'intérêt de l'interprétabilité pour le décideur

---

## ✅ CHECKLIST PARTIE 7 – Rédaction du Rapport

### Structure minimale du rapport

- [ ] **Page de garde** : titre, auteurs, date
- [ ] **Introduction** : rappel de la classification partielle et de l'objectif
- [ ] **Section 1 – Modélisation** : définition du problème MO (solution, objectifs, représentation, opérateurs)
- [ ] **Section 2 – Algorithmes** : description de NSGA-II et de l'autre AG choisi
- [ ] **Section 3 – Protocole expérimental** : datasets, splits, métriques, nb de runs
- [ ] **Section 4 – Tuning** : tableaux, boxplots, conclusion sur les paramètres
- [ ] **Section 5 – Comparaison des AG** : boxplots, test statistique, conclusion
- [ ] **Section 6 – Comparaison Sklearn** : Figure 1, Figure 2, Figure 3, tableau récapitulatif
- [ ] **Section 7 – Analyse et Interprétabilité** : discussion, règles extraites
- [ ] **Conclusion** : synthèse des résultats et perspectives

### Éléments visuels à inclure

- [ ] Boxplots du tuning (HV vs pop_size) pour chaque algo et dataset
- [ ] Boxplot de comparaison NSGA-II vs Autre AG
- [ ] Figure 1 : 20 fronts Pareto (un par run) + meilleur front mis en évidence
- [ ] Figure 2 : Meilleur front Pareto + points C4.5, RF, SVM
- [ ] Figure 3 : Solution choisie (meilleur F1) mise en évidence sur le front

---

## ✅ CHECKLIST PARTIE 8 – Rendu final

- [ ] Vérifier que le rapport répond à **toutes les questions** du sujet
- [ ] Vérifier que chaque tableau de résultats est complet (training + test)
- [ ] Vérifier que le code est bien commenté
- [ ] Vérifier que le code est exécutable (dépendances documentées)
- [ ] Créer un fichier `requirements.txt` :
  ```
  jmetalpy
  scikit-learn
  matplotlib
  pandas
  numpy
  scipy
  ```
- [ ] Créer le fichier `.zip` contenant :
  - [ ] `rapport.pdf`
  - [ ] Dossier `src/` avec tous les scripts Python
  - [ ] Dossier `results/` avec les résultats générés
  - [ ] `README.md`
  - [ ] `requirements.txt`
- [ ] Vérifier le nom du fichier zip : `NOM1_NOM2_TP5.zip`
- [ ] **Soumettre le rendu** avant la deadline

---

## 📌 Métriques clés à connaître

| Métrique | Formule | Objectif |
|---|---|---|
| **Confiance (Precision)** | TP / (TP + FP) | Maximiser |
| **Sensibilité (Recall)** | TP / (TP + FN) | Maximiser |
| **F1-mesure** | 2 × (P × R) / (P + R) | Référence |
| **Hypervolume (HV)** | Volume dominé par le front | Maximiser |

---

## 🔧 Commandes utiles

```bash
# Installer les dépendances
pip install jmetalpy scikit-learn matplotlib pandas numpy scipy

# Lancer une exécution NSGA-II
python src/nsga2_runner.py --dataset yeast1 --pop_size 100 --runs 20

# Lancer le tuning
python src/tuning.py --dataset pima_diabete

# Générer les graphiques
python src/analysis.py --output results/
```

---

## ⚠️ Points d'attention importants

> 1. **JMetalPy minimise** les objectifs → toujours inverser le signe pour maximiser (`-precision`, `-recall`)
>
> 2. **Hypervolume** : définir un **point de référence** fixe pour tous les calculs HV (ex: `[0, 0]` si les objectifs sont dans `[-1, 0]`)
>
> 3. **20 runs minimum** par configuration testée — ne pas oublier de fixer la seed différemment à chaque run
>
> 4. **Training vs Test** : les fronts Pareto sont construits sur le training, mais l'évaluation finale se fait sur le test
>
> 5. **Expliquer clairement** comment un seul score est extrait des 20 fronts Pareto pour remplir les tableaux comparatifs

---

*README généré pour le projet TP5 – M2 MIAGE, OAFD, Université de Lille*