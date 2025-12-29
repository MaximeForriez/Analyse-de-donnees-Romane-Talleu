#coding:utf8

# Sources des données : production de M. Forriez, 2016-2023
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("img", exist_ok=True)

with open("data/resultats-elections-presidentielles-2022-1er-tour.csv", "r", encoding="utf-8") as f:
    df = pd.read_csv("data/resultats-elections-presidentielles-2022-1er-tour.csv", encoding="utf-8")
print(df.head())


colonnes_quantitatives = df.select_dtypes(include=np.number).columns.tolist()
print("Colonnes quantitatives :", colonnes_quantitatives)

moyennes = df[colonnes_quantitatives].mean().round(2).tolist()
medianes = df[colonnes_quantitatives].median().round(2).tolist()
modes = [df[col].mode()[0] for col in colonnes_quantitatives]
ecarts_types = df[colonnes_quantitatives].std().round(2).tolist()
ecarts_absolus = df[colonnes_quantitatives].apply(lambda x: np.abs(x - x.mean()).mean()).round(2).tolist()
etendues = (df[colonnes_quantitatives].max() - df[colonnes_quantitatives].min()).round(2).tolist()

print("Moyennes :", moyennes)
print("Médianes :", medianes)
print("Modes :", modes)
print("Écarts-types :", ecarts_types)
print("Écarts absolus à la moyenne :", ecarts_absolus)
print("Étendues :", etendues)

iqr = (df[colonnes_quantitatives].quantile(0.75) - df[colonnes_quantitatives].quantile(0.25)).round(2).tolist()
idr = (df[colonnes_quantitatives].quantile(0.9) - df[colonnes_quantitatives].quantile(0.1)).round(2).tolist()

print("Distance interquartile :", iqr)
print("Distance interdécile :", idr)

for col in colonnes_quantitatives:
    plt.figure()
    df[col].plot.box()
    plt.title(col)
    plt.savefig(f"img/{col}_boxplot.png")
    plt.close()

islands = pd.read_csv("data/island-index.csv", encoding="utf-8")
surface = islands["Surface (km²)"]

bins = [0, 10, 25, 50, 100, 2500, 5000, 10000, np.inf]
labels = ["]0,10]", "]10,25]", "]25,50]", "]50,100]", "]100,2500]", "]2500,5000]", "]5000,10000]", "]10000,+∞["]

surface_categorie = pd.cut(surface, bins=bins, labels=labels, right=True)
compte_categorie = surface_categorie.value_counts().sort_index()

print("Nombre d'îles par catégorie de surface :")
print(compte_categorie)

resultats = pd.DataFrame({
    "Moyennes": moyennes,
    "Médianes": medianes,
    "Modes": modes,
    "Écarts-types": ecarts_types,
    "Écarts absolus": ecarts_absolus,
    "Étendues": etendues,
    "IQR": iqr,
    "IDR": idr
}, index=colonnes_quantitatives)

resultats.to_csv("data/statistiques.csv", index=True)
resultats.to_excel("data/statistiques.xlsx", index=True)

print("=== Traitement terminé, résultats exportés ===")
