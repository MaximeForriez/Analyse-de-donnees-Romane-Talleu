# coding: utf8

import math
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


def ouvrir_un_fichier(nom: str) -> pd.DataFrame:
    return pd.read_csv(nom, encoding="utf-8")


def conversion_log(liste: List[float]) -> List[float]:
    out = []
    for element in liste:
        try:
            val = float(element)
            out.append(math.log(val) if val > 0 else float("nan"))
        except Exception:
            out.append(float("nan"))
    return out


def ordre_decroissant(liste: List[float]) -> List[float]:
    copie = list(liste)
    copie.sort(reverse=True)
    return copie


def ordre_population(pop: List[float], etat: List[str]) -> List[Tuple[int, str]]:
    ordrepop: List[Tuple[float, str]] = []
    for i in range(len(pop)):
        try:
            val = float(pop[i])
        except Exception:
            val = float("nan")
        if not np.isnan(val):
            ordrepop.append((val, etat[i]))
    ordrepop_sorted = sorted(ordrepop, key=lambda x: x[0], reverse=True)

    resultat: List[Tuple[int, str]] = []
    for idx, item in enumerate(ordrepop_sorted):
        resultat.append((idx + 1, item[1]))
    return resultat


def classement_pays(ordre1: List[Tuple[int, str]], ordre2: List[Tuple[int, str]]) -> List[Tuple[int, int, str]]:
    map1 = {etat: rang for rang, etat in ordre1}
    map2 = {etat: rang for rang, etat in ordre2}
    classement: List[Tuple[int, int, str]] = []

    for etat in map1:
        if etat in map2:
            classement.append((map1[etat], map2[etat], etat))
    classement.sort(key=lambda x: x[0])
    return classement


def clean_numeric_series(s: List) -> List[float]:
    def clean_val(v):
        if pd.isna(v):
            return float("nan")
        st = str(v)
        st = st.replace("\xa0", " ")
        st = re.sub(r"[a-zA-Zµ°/]+", "", st)
        st = st.replace(",", ".")
        st = re.sub(r"\s+", "", st)
        st = re.sub(r"[^\d\.\-]", "", st)
        if st in ("", ".", "-", "-."):
            return float("nan")
        try:
            return float(st)
        except Exception:
            return float("nan")

    return [clean_val(x) for x in s]


def spearman_kendall(list1: List[float], list2: List[float]) -> Tuple[float, float, float, float]:
    a = np.array(list1, dtype=float)
    b = np.array(list2, dtype=float)
    mask = ~ (np.isnan(a) | np.isnan(b))
    a2 = a[mask]
    b2 = b[mask]
    if len(a2) < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    rs = scipy.stats.spearmanr(a2, b2)
    kt = scipy.stats.kendalltau(a2, b2)
    return (float(rs.correlation), float(rs.pvalue), float(kt.correlation), float(kt.pvalue))


def compare_rankings(rang1_values: List[float], rang2_values: List[float]) -> Dict[str, float]:
    sp_cor, sp_p, kd_cor, kd_p = spearman_kendall(rang1_values, rang2_values)
    return {
        "spearman_corr": sp_cor,
        "spearman_p": sp_p,
        "kendall_corr": kd_cor,
        "kendall_p": kd_p,
    }


def safe_plot(x: List[float], y: List[float], xlabel: str, ylabel: str, title: str,
              save_path: Optional[Path] = None) -> Optional[Path]:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return None


def analyze_all_years_rank_concordance(df: pd.DataFrame, etat_col: str, pop_prefix: str = 'Pop ') -> Optional[pd.DataFrame]:
    years = []
    for col in df.columns:
        if isinstance(col, str) and col.strip().startswith(pop_prefix.strip()):
            m = re.search(r"(\d{4})", col)
            if m:
                years.append(int(m.group(1)))
    years = sorted(set(years))
    if not years:
        return None

    rankings = {}
    etats = df[etat_col].astype(str).tolist()
    for y in years:
        colname = f"{pop_prefix}{y}"
        if colname not in df.columns:
            continue
        values = clean_numeric_series(df[colname])
        ordpop = ordre_population(values, etats)
        maprank = {etat: r for r, etat in ordpop}
        ranks_list = [maprank.get(e, float('nan')) for e in etats]
        rankings[y] = ranks_list

    years_list = sorted(rankings.keys())
    n = len(years_list)
    tau_mat = pd.DataFrame(index=years_list, columns=years_list, dtype=float)
    for i, y1 in enumerate(years_list):
        for j, y2 in enumerate(years_list):
            if i <= j:
                a = rankings[y1]
                b = rankings[y2]
                tau, _p = scipy.stats.kendalltau(
                    np.array(a, dtype=float),
                    np.array(b, dtype=float),
                    nan_policy='omit'
                )
                tau_mat.loc[y1, y2] = float(tau) if not pd.isna(tau) else float('nan')
                tau_mat.loc[y2, y1] = tau_mat.loc[y1, y2]
    return tau_mat


def main(data_dir: str = "./data",
         save_outputs: bool = False,
         verbose: bool = False) -> Dict[str, object]:
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Le dossier {data_dir} est introuvable.")

    results: Dict[str, object] = {}
    df_iles = ouvrir_un_fichier(data_path / "island-index.csv")
    col_surface_name = None
    for col in df_iles.columns:
        if 'surface' in str(col).lower():
            col_surface_name = col
            break
    if col_surface_name is None:
        raise KeyError("Impossible de trouver la colonne 'Surface' dans island-index.csv.")

    surfaces_raw = df_iles[col_surface_name].tolist()
    surfaces = clean_numeric_series(surfaces_raw)
    continents = [85545323.0, 37856841.0, 7768030.0, 7605049.0]
    surfaces.extend(continents)

    surfaces = [float(x) if not pd.isna(x) else float('nan') for x in surfaces]
    surfaces_nonan = [x for x in surfaces if not math.isnan(x)]
    surfaces_ord = ordre_decroissant(surfaces_nonan)
    rangs = list(range(1, len(surfaces_ord) + 1))

    saved_plots: Dict[str, Path] = {}
    if save_outputs:
        p1 = safe_plot(rangs, surfaces_ord,
                       xlabel="Rang",
                       ylabel="Surface (km2)",
                       title="Loi rang-taille - surfaces (échelle linéaire)",
                       save_path=Path("./rang_taille_linear.png"))
        p2 = safe_plot(conversion_log(rangs), conversion_log(surfaces_ord),
                       xlabel="log(Rang)",
                       ylabel="log(Surface (km2))",
                       title="Loi rang-taille - surfaces (log-log)",
                       save_path=Path("./rang_taille_loglog.png"))
        if p1:
            saved_plots['rang_taille_linear'] = p1
        if p2:
            saved_plots['rang_taille_loglog'] = p2

    results['surfaces_ord'] = surfaces_ord
    results['surfaces_ranks'] = rangs
    results['saved_plots'] = saved_plots

    df_monde = ouvrir_un_fichier(data_path / "Le-Monde-HS-Etats-du-monde-2007-2025.csv")

    etat_col = None
    for col in df_monde.columns:
        if 'etat' in str(col).lower() or 'état' in str(col).lower():
            etat_col = col
            break
    if etat_col is None:
        raise KeyError("Impossible de trouver la colonne 'État' dans le fichier Le-Monde...")
    pop2007_col = None
    pop2025_col = None
    dens2007_col = None
    dens2025_col = None
    for col in df_monde.columns:
        low = str(col).lower()
        if 'pop' in low and '2007' in low:
            pop2007_col = col
        if 'pop' in low and '2025' in low:
            pop2025_col = col
        if 'dens' in low and '2007' in low:
            dens2007_col = col
        if 'dens' in low and '2025' in low:
            dens2025_col = col

    if pop2007_col is None:
        for col in df_monde.columns:
            if '2007' in str(col) and any(w in str(col).lower() for w in ('pop', 'population')):
                pop2007_col = col
                break
    if pop2025_col is None:
        for col in df_monde.columns:
            if '2025' in str(col) and any(w in str(col).lower() for w in ('pop', 'population')):
                pop2025_col = col
                break
    if dens2007_col is None:
        for col in df_monde.columns:
            if '2007' in str(col) and any(w in str(col).lower() for w in ('dens', 'density')):
                dens2007_col = col
                break
    if dens2025_col is None:
        for col in df_monde.columns:
            if '2025' in str(col) and any(w in str(col).lower() for w in ('dens', 'density')):
                dens2025_col = col
                break

    etats = df_monde[etat_col].astype(str).tolist()
    pop2007 = clean_numeric_series(df_monde[pop2007_col]) if pop2007_col else [float('nan')] * len(etats)
    pop2025 = clean_numeric_series(df_monde[pop2025_col]) if pop2025_col else [float('nan')] * len(etats)
    dens2007 = clean_numeric_series(df_monde[dens2007_col]) if dens2007_col else [float('nan')] * len(etats)
    dens2025 = clean_numeric_series(df_monde[dens2025_col]) if dens2025_col else [float('nan')] * len(etats)

    ord_pop2007 = ordre_population(pop2007, etats)
    ord_pop2025 = ordre_population(pop2025, etats)
    ord_dens2007 = ordre_population(dens2007, etats)
    ord_dens2025 = ordre_population(dens2025, etats)

    classement_2007_pop_vs_dens = classement_pays(ord_pop2007, ord_dens2007)
    results['classement_2007_pop_vs_dens'] = classement_2007_pop_vs_dens

    rangs_pop_2007 = [item[0] for item in classement_2007_pop_vs_dens]
    rangs_dens_2007 = [item[1] for item in classement_2007_pop_vs_dens]
    stats_2007 = compare_rankings(rangs_pop_2007, rangs_dens_2007)
    results['stats_2007_pop_vs_dens'] = stats_2007

    if save_outputs and classement_2007_pop_vs_dens:
        df_cl_2007 = pd.DataFrame(classement_2007_pop_vs_dens, columns=['rang_pop_2007', 'rang_dens_2007', 'etat'])
        df_cl_2007.to_csv("classement_pop_vs_dens_2007.csv", index=False, encoding='utf-8')
        results['classement_2007_df'] = df_cl_2007

    tau_matrix = analyze_all_years_rank_concordance(df_monde, etat_col, pop_prefix='Pop ')
    results['kendall_tau_matrix_population_years'] = tau_matrix

    if save_outputs and tau_matrix is not None:
        tau_matrix.to_csv("kendall_tau_matrix_population_years_2007_2025.csv", encoding='utf-8')

    if verbose:

  return results
