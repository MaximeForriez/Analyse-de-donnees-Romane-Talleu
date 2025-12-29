# coding:utf8

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

os.makedirs("s4", exist_ok=True)

dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']
print(dist_names)

def dirac_pmf(a, support):
    pmf = np.zeros_like(support, dtype=float)
    pmf[np.where(support == a)] = 1.0
    return pmf

def plot_discrete_pmf(x, pmf, nom_fichier, titre):
    plt.figure(figsize=(6,3.5))
    plt.stem(x, pmf)  
    plt.title(titre)
    plt.xlabel("Valeurs")
    plt.ylabel("P(X=k)")
    plt.grid(True)
    plt.savefig(f"s4/{nom_fichier}.png")
    plt.close()

def plot_continuous_pdf(x, pdf, nom_fichier, titre):
    plt.figure(figsize=(6,3.5))
    plt.plot(x, pdf)
    plt.title(titre)
    plt.xlabel("Valeurs")
    plt.ylabel("Densité de probabilité f(x)")
    plt.grid(True)
    plt.savefig(f"s4/{nom_fichier}.png")
    plt.close()

def moyenne(data):
    return np.mean(data)

def ecart_type(data):
    return np.std(data)

support = np.arange(-1, 6)
pmf = dirac_pmf(2, support)
plot_discrete_pmf(support, pmf, "loi_dirac", "Loi de Dirac (a=2)")

n = 6
x = np.arange(1, n+1)
pmf = np.ones_like(x)/n
plot_discrete_pmf(x, pmf, "loi_uniforme_discrete", "Loi uniforme discrète [1..6]")

n, p = 10, 0.5
x = np.arange(0, n+1)
pmf = stats.binom.pmf(x, n, p)
plot_discrete_pmf(x, pmf, "loi_binomiale", "Loi binomiale (n=10, p=0.5)")

lam = 4
x = np.arange(0, 15)
pmf = stats.poisson.pmf(x, mu=lam)
plot_discrete_pmf(x, pmf, "loi_poisson", "Loi de Poisson (λ=4)")

def zipf_mandelbrot_pmf(k_arr, a=0.5, b=1.0, c=1.2):
    raw = 1.0 / (a + b*k_arr)**c
    return raw / raw.sum()

k = np.arange(1, 20)
pmf = zipf_mandelbrot_pmf(k)
plot_discrete_pmf(k, pmf, "loi_zipf", "Loi de Zipf-Mandelbrot (a=0.5, b=1, c=1.2)")

x = np.linspace(-4, 4, 200)
pdf = stats.norm.pdf(x, loc=0, scale=1)
plot_continuous_pdf(x, pdf, "loi_normale", "Loi normale (μ=0, σ=1)")

x = np.linspace(0, 5, 200)
pdf = stats.lognorm.pdf(x, s=0.5)
plot_continuous_pdf(x, pdf, "loi_lognormale", "Loi log-normale (σ=0.5)")

x = np.linspace(0, 1, 200)
pdf = stats.uniform.pdf(x, loc=0, scale=1)
plot_continuous_pdf(x, pdf, "loi_uniforme_continue", "Loi uniforme continue [0,1]")

x = np.linspace(0, 10, 200)
pdf = stats.chi2.pdf(x, df=3)
plot_continuous_pdf(x, pdf, "loi_chi2", "Loi du χ² (ddl=3)")

x = np.linspace(1, 10, 200)
pdf = stats.pareto.pdf(x, b=2)
plot_continuous_pdf(x, pdf, "loi_pareto", "Loi de Pareto (b=2)")

print("\n--- Moyenne et écart-type (calculés avec scipy.stats) ---")
distributions = {
    "Binomiale": stats.binom(10, 0.5),
    "Poisson": stats.poisson(4),
    "Normale": stats.norm(0, 1),
    "Log-normale": stats.lognorm(0.5),
    "Uniforme continue": stats.uniform(0, 1),
    "Chi²": stats.chi2(3),
    "Pareto": stats.pareto(2)
}

for nom, dist in distributions.items():
    print(f"{nom:18s} → moyenne = {dist.mean():.3f}, écart-type = {dist.std():.3f}")

echantillon = stats.norm.rvs(size=1000)
print("\nExemple (échantillon normal simulé) :")
print("Moyenne =", moyenne(echantillon))
print("Écart-type =", ecart_type(echantillon))


