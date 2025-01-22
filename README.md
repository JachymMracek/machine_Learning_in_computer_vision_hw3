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
#### DecisionTree_: {'criterion': 'entropy'}
#### MLPClassifier_: {'activation': 'tanh'}
#### GradientBoosting_: {'max_depth': 5,'subsample': 0.8}


## TASK 4
Použil jsem k-means metodu, kde optimální počet clusterů jsem získal pomocí elbow metody, kde ze získaných intertia jsem vypočítal vzdálenost od přímky náležící počátečnímu bodu a koncovému bodu grafu. Maximální vzdálenost bodu od naší přímky je náš elbow bod.
Normálový tvar rovnice naší přímky je ax + by + c = 0, kde smerovy_vektor = (-b,a), kde -b = len(WSS) - 1, uprávou b = - ( len(WSS) - 1), a =  WSS[-1] - WSS[0], c = -(a*1 + b* WSS[0]).
Kde vzdálenost od našeho bodu (x,y) je d = abs(a*x + b*y + c) / np.sqrt(np.square(a)+np.square(b))
![image](https://github.com/user-attachments/assets/8da88855-37f0-4e17-bffb-f40383befb46)


## TASK 5
Komplikací bylo dlouhé zkoušení parametrů a čekání na výpočet grid sreach a proto jsou parametry vybrány, tak aby grid search trval rozumně dlouho.
Výběr modelů byl založen na přesnosti defaultních parametrů příslušných modelů na celém treninkovém datasetu, kde právě tyto tři modely vyšli, jako nejepší. Například s porovnámín Naivního bayese nebo logistické regrese.
Parametry pro grid search obsahují vždy defaultní hodnotu a alespoń jednu hodnotu větší a menší, jak defaultní hodnota. Byly vybrány nejduležitější parametry.

## TASK 6
Spuštěním evaluate uživatel dostane na standartím výstupu kovarianční matici,presion a recall ve formátu:

GradientBoosting

|     | 1   | 2   | 3   | 4   | 5   | 6   | 7   |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 1   | 291 | 0   | 2   | 0   | 0   | 0   | 7   |
| 2   | 2   | 289 | 1   | 2   | 0   | 0   | 6   |
| 3   | 5   | 1   | 289 | 0   | 0   | 0   | 5   |
| 4   | 0   | 3   | 0   | 297 | 0   | 0   | 0   |
| 5   | 0   | 0   | 0   | 0   | 300 | 0   | 0   |
| 6   | 0   | 0   | 0   | 0   | 0   | 300 | 0   |
| 7   | 11  | 0   | 4   | 1   | 0   | 0   | 284 |



precision: 0.9764063425753082

recall: 0.9761904761904762

![image](https://github.com/user-attachments/assets/250e1941-ff93-4d76-9d8e-551a3925509d)


Uživatel také dostane výsledky modelů v presion-recall prostoru. ( Graf je pro grid search parametry a evaluate_on_train je true)
