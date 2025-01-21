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
Použil jsem k-means metodu, kde optimální počet clusterů jsem získal pomocí elbow metody, kde ze získaných intertia jsem vypočítal vzdálenost od přímky náležící počátečnímu bodu a koncovému bodu grafu. Maximální vzdálenost od bodu je náš elbow bod. Vzorec pro vzdálenost je:
\[
d = \frac{|(WSS[-1] - WSS[0]) \cdot (i+1) + (1 - n) \cdot WSS[i] + (n \cdot WSS[0] - WSS[-1])|}{\sqrt{(WSS[-1] - WSS[0])^2 + (1 - n)^2}}
\]
