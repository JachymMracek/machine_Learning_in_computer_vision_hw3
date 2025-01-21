# HW3
## TASK 1
Seznamil jsem se s datasetem a využil attributy v kodu.

## TASK 2
Byla provedena analýza a pomocí PCA jsem redukoval dimenzi, čímž jsem provedli feature selection. Vykreslený graf závislosti rozptylu na n_components je:
![image](https://github.com/user-attachments/assets/a376c843-6cdd-47f2-8d75-4f989b9ab005)

Optimální n_components = 4.

## TASK 3
Vybral jsem tři supervised metody -> rozhodovací stromy, neuronové sítě a gradient boosting metodu. Výběr parametrů jsem provedl pomocí metody grid search a pomocí cross-validace. Následně jsem natrénoval modely na získaných parametrech a uložil modely.

### Výsledky grid search:
#### DecisionTree_: {'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
#### MLPClassifier_: {'activation': 'tanh', 'alpha': 1e-05, 'early_stopping': False, 'hidden_layer_sizes': (150,), 'learning_rate_init': 0.001}
#### GradientBoosting_: {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 0.8}

### Pokud uživatel nechce grid search:
#### DecisionTree_: {}
#### MLPClassifier_: {'activation': 'tanh'}
#### GradientBoosting_: {'max_depth': 5,'subsample': 0.8}


## TASK 4
Použil jsem k-means metodu, kde optimální počet clusterů jsem získal pomocí elbow metody, kde ze získaných intertia jsem vypočítal vzdálenost od přímky náležící počátečnímu bodu a koncovému bodu grafu. Maximální vzdálenost bodu od naší přímky je náš elbow bod.
![image](https://github.com/user-attachments/assets/25ce1234-8cbe-4246-8f58-9dbb6a64df34)

## TASK 5
Program trénuje modely, které mají nastavené parametry defaultně, pokud uživatel nechce nejleší volbu parametrů.
Komplikací bylo dlouhé zkoušení parametrů a čekání na výpočet grid sreach a proto jsou parametry vybrány, tak aby grid search trval rozumně dlouho.
Výběr modelů byl založen na přesnosti defaultních parametrů příslušných modelů, kde právě tyto tři modely vyšli, jako nejepší. Například s porovnámín Naivního bayese nebo logistické regrese.
Parametry pro grid search obsahují vždy defaultní hodnotu a alespoń jednu hodnotu větší a menší, jak defaultní hodnota. Byly vybrány nejduležitější parametry.

## TASK 6
Spuštěním evaluate uživatel dostane na standartím výstupu kovarianční matici,presion a recall ve formátu:

DecisionTree

|     |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
|-----|-----|-----|-----|-----|-----|-----|-----|
| **0** | 300 |   0 |   0 |   0 |   0 |   0 |   0 |
| **1** |   0 | 300 |   0 |   0 |   0 |   0 |   0 |
| **2** |   0 |   0 | 300 |   0 |   0 |   0 |   0 |
| **3** |   0 |   0 |   0 | 300 |   0 |   0 |   0 |
| **4** |   0 |   0 |   0 |   0 | 300 |   0 |   0 |
| **5** |   0 |   0 |   0 |   0 |   0 | 300 |   0 |
| **6** |   0 |   0 |   0 |   0 |   0 |   0 | 300 |

precision: 1.0

recall: 1.0

![image](https://github.com/user-attachments/assets/6e9d1d9c-3630-432a-8aaf-d0731dbb01df)

Uživatel také dostane výsledky modelů v presion-recall prostoru. (GRAF JE PRO DEFAULTNÍ PARAMETRY MODELŮ)
