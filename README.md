# HW3 README
## TASK 1
Seznamil jsem se s datasetem a využil attributy v kodu.

## TASK 2
Byla proveden analýza a pomocí PCA jsem redukoval dimenzi, čímž jsem provedli feature selection. Vykreslený graf závislosti rozptylu na n_components je:
![image](https://github.com/user-attachments/assets/a376c843-6cdd-47f2-8d75-4f989b9ab005)

Optimální n_components = 4.

## TASK 3
Vybral jsem tři supervised metody -> rozhodovací stromy, neuronové sítě a gradient boosting metodu. Výběr parametrů jsem provedl pomocí metody grid search a pomocí cross-validace. Následně jsem natrénoval modely na získaných parametrech a uložil modely.

### Výsledky grid search:
#### DecisionTree_: {'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
#### MLPClassifier_: {'activation': 'tanh', 'alpha': 0.001, 'batch_size': 'auto', 'early_stopping': True, 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.01, 'max_iter': 500, 'solver': 'adam'}
#### GradientBoosting_: {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 0.8}

## TASK 4
Použil jsem k-means metodu, kde optimální počet clusterů jsem získal pomocí elbow metody, kde ze získaných intertia jsem vypočítal vzdálenost od přímky náležící počátečnímu bodu a koncovému bodu grafu. Maximální vzdálenost bodu od naší přímky je náš elbow bod.
![image](https://github.com/user-attachments/assets/25ce1234-8cbe-4246-8f58-9dbb6a64df34)

## TASK 5
Parametry modelu jsou vybrány právě pomocí grid search, jelikož jsou optimální pro naší úlohu a modely byly vybrány podle přesnosti na našich datech.
Komplikací úlohy bylo dlouhé zkoušení parametrů a čekání na výpočet grid sreach.

## TASK 6
Spuštěním evaluate uživatel dostane na standartím výstupu kovarianční matici,presion a recall ve formátu:

DecisionTree

|   254 |     0 |     0 |     0 |     0 |     0 |    46 |
|-------|-------|-------|-------|-------|-------|-------|
|    30 |   231 |     3 |    10 |     6 |     0 |    20 |
|   208 |     9 |    34 |     0 |     0 |     0 |    49 |
|     0 |     0 |     0 |   284 |     0 |     0 |    16 |
|     0 |     0 |     5 |     4 |   291 |     0 |     0 |
|     0 |     0 |     0 |     0 |     0 |   300 |     0 |
|   107 |     6 |     6 |     0 |     0 |     0 |   181 |

precision: 0.7977634442156533

recall: 0.75

![image](https://github.com/user-attachments/assets/5ec2a9df-2bda-42d2-9072-7fc58f66f10b)

Uživatel také dostane výsledky modelů v presion-recall prostoru.
