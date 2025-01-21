# HW3 README
## TASK 1
Seznamil jsem se s datasetem a využil attributy v kodu

## TASK 2
Provedl jsem analýzu příznaků a sestrojil graf závislosti rozptylu na počtu komponent a následně jsem zadrátoval nejlepší hodnotu do programu, aby ostatní volání funkcí proběhlo v pořádku. n_componets jsem zvolil na hodonotu 4, čímž proběhů feature selection a snížil
jsem pomocí PCA dimenzi
![image](https://github.com/user-attachments/assets/a376c843-6cdd-47f2-8d75-4f989b9ab005)

## TASK 3
Vybral jsem tři supervised metody rozhodovací stromy, logistickou regresi a naivní Bayes. Výběr parametrů jsem provedl pomocí metody grid search a pomocí cross-validace. Následně jsem natrénoval modely na získaných parametrech a uložil modely.

## TASK 4
Použil jsem k-means metodu, kde optimální počet clusterů jsem získal pomocí elbow metody, kde ze získaných intertia jsem vypočítal vzdálenost od přímky náležící počátečnímu bodu a koncovému bodu grafu. Maximální vzdálenost bodu od naší přímkx je náš elbow bod.
![image](https://github.com/user-attachments/assets/25ce1234-8cbe-4246-8f58-9dbb6a64df34)

## TASK 5
Parametry modelu jsou vybrány právě pomocí grid search, jelikož jsou optimální pro naší úlohu.

## TASK 6

Spuštěním evaluate uživatel dostane na standartím výstupu kovarianční matici,presion a recall ve formátu:

NaiveBayes

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
