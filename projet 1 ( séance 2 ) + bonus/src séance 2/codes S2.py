#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt

with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(fichier)

import pandas as pd

chemin_fichier = "data/resultats-elections-presidentielles-2022-1er-tour.csv"

with open(chemin_fichier, "r") as fichier:
    contenu = pd.read_csv(fichier, sep=',', quotechar='"')

print("=== Aperçu du contenu du CSV ===")
print(contenu)

print("\n=== Dimensions du tableau ===")
print(f"Lignes : {len(contenu)}")
print(f"Colonnes : {len(contenu.columns)}")

print("\n=== Types des colonnes ===")
types_colonnes = {col: contenu[col].dtype for col in contenu.columns}
for col, type_ in types_colonnes.items():
    print(f"{col} : {type_}")
print("\n=== Noms des colonnes ===")
print(contenu.head(0)) 
print("\n=== Colonne 'Inscrits' ===")
if 'Inscrits' in contenu.columns:
    print(contenu['Inscrits'])
else:
    print("⚠️ La colonne 'Inscrits' n'existe pas (vérifie le nom exact avec head())")
print("\n=== Sommes des colonnes numériques ===")
for col in contenu.columns:
    if pd.api.types.is_numeric_dtype(contenu[col]):
        print(f"{col} : {contenu[col].sum()}")
        import os
import matplotlib.pyplot as plt
os.makedirs("images/barres", exist_ok=True)

for i, ligne in contenu.iterrows():
    departement = ligne["Libellé du département"]
    inscrits = ligne["Inscrits"]
    votants = ligne["Votants"]

    plt.figure(figsize=(5, 3))
    plt.bar(["Inscrits", "Votants"], [inscrits, votants])
    plt.title(f"{departement} - Inscrits vs Votants")
    plt.ylabel("Nombre de personnes")

    plt.tight_layout()
    plt.savefig(f"images/barres/{departement.replace('/', '_')}.png")
    plt.close()

print("✅ Diagrammes en barres créés dans le dossier images/barres/")

os.makedirs("images/camemberts", exist_ok=True)

for i, ligne in contenu.iterrows():
    departement = ligne["Libellé du département"]
    blancs = ligne["Blancs"]
    nuls = ligne["Nuls"]
    exprimes = ligne["Exprimés"]
    abstentions = ligne["Abstentions"]

    valeurs = [blancs, nuls, exprimes, abstentions]
    labels = ["Blancs", "Nuls", "Exprimés", "Abstentions"]

    plt.figure(figsize=(4, 4))
    plt.pie(valeurs, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(f"{departement} - Répartition des votes")

    plt.tight_layout()
    plt.savefig(f"images/camemberts/{departement.replace('/', '_')}.png")
    plt.close()

print("✅ Diagrammes circulaires créés dans le dossier images/camemberts/")

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.hist(contenu["Inscrits"], bins=20, density=True, edgecolor="black")
plt.title("Distribution du nombre d'inscrits")
plt.xlabel("Nombre d'inscrits")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig("images/histogramme_inscrits.png")
plt.close()

print("✅ Histogramme des inscrits créé dans images/histogramme_inscrits.png")

