# -*- coding: utf-8 -*-
"""Допоміжні функції для аналізу та побудови діаграм спрямованості (ДС).

Використовуються в усіх лабораторних роботах з дисципліни
«Антенні пристрої та комплекси».
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 11,
    "mathtext.fontset": "dejavuserif",
})


def analyze(theta_deg, F, main_dir=None):
    """Аналіз нормованої ДС |F(θ)|.

    Повертає словник:
      main  – напрямок головного максимуму, град;
      left, right – кути, на яких |F| = 0,707 ліворуч і праворуч від максимуму;
      hpbw  – ширина головної пелюстки на рівні 0,707, град;
      nulls – напрямки нульового (мінімального) випромінювання, град;
      lobes – список (кут, рівень) максимумів бічних пелюсток.
    """
    th = np.asarray(theta_deg, float)
    a = np.abs(np.asarray(F, float))
    a = a / a.max()
    i0 = int(np.argmax(a)) if main_dir is None else int(np.argmin(np.abs(th - main_dir)))
    lvl = 1 / math.sqrt(2)

    def cross(direction):
        i = i0
        while a[i] > lvl and 0 <= i + direction < len(a):
            i += direction
        if a[i] > lvl or i == i0:
            return None
        j = i - direction
        return float(np.interp(lvl, [a[i], a[j]], [th[i], th[j]]))

    left, right = cross(-1), cross(+1)
    if left is None and right is not None and abs(th[0]) < 1e-9:
        left = -right  # симетрична ДС, побудована на [0, 90°]
    hpbw = (right - left) if (left is not None and right is not None) else None
    nulls, lobes = [], []
    for i in range(1, len(a) - 1):
        if a[i] <= a[i - 1] and a[i] < a[i + 1] and a[i] < 0.5:
            nulls.append(float(th[i]))
        if a[i] >= a[i - 1] and a[i] > a[i + 1] and i != i0 and a[i] < 0.999:
            lobes.append((float(th[i]), float(a[i])))
    return {"main": float(th[i0]), "left": left, "right": right, "hpbw": hpbw,
            "nulls": nulls, "lobes": lobes, "sll_db": 20 * math.log10(max(l[1] for l in lobes)) if lobes else None}


def plot_linear(fname, theta, curves, res, title=None, xlim=(0, 90), ylabel=r"$|F(\theta)|$",
                xlabel=r"$\theta$, град", mark_key=None, step=10, annotate=True):
    """Графік ДС у прямокутній системі координат з позначенням рівня 0,707, нулів і бічних пелюсток.

    curves – список (масив F, підпис, стиль); res – словник з результатами analyze() для кривих,
    на яких треба зробити позначки: {підпис: res}."""
    fig, ax = plt.subplots(figsize=(17 / 2.54, 10 / 2.54))
    for F, label, style in curves:
        ax.plot(theta, np.abs(F), style, lw=1.2, label=label)
    ax.axhline(1 / math.sqrt(2), color="0.4", ls=":", lw=0.8)
    ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), 0.72, "0,707", fontsize=8, color="0.3")
    markers = [("o", "C3"), ("s", "C2"), ("^", "C4"), ("D", "C5")]
    for k, (lab, r) in enumerate(res.items()):
        mk, col = markers[k % len(markers)]
        if r["left"] is not None and r["right"] is not None:
            for x in (r["left"], r["right"]):
                if xlim[0] <= x <= xlim[1]:
                    ax.plot([x, x], [0, 1 / math.sqrt(2)], "--", color=col, lw=0.7)
            ax.plot([max(r["left"], xlim[0]), min(r["right"], xlim[1])], [1 / math.sqrt(2)] * 2, "-", color=col, lw=1.6,
                    label=f"2θ0,5 {lab} = {r['hpbw']:.2f}°".replace(".", ","))
        nx = [x for x in r["nulls"] if xlim[0] <= x <= xlim[1]]
        ax.plot(nx, [0] * len(nx), "x", color=col, ms=7, mew=1.5, label=f"нулі {lab}")
        lx = [(x, y) for x, y in r["lobes"] if xlim[0] <= x <= xlim[1]]
        if lx:
            ax.plot([x for x, _ in lx], [y for _, y in lx], mk, color=col, ms=4.5, mfc="none", mew=1.2,
                    label=f"бічні пелюстки {lab}")
            if annotate:
                for x, y in lx[:8]:
                    ax.annotate(f"{y:.3f}".replace(".", ","), (x, y), textcoords="offset points", xytext=(2, 5),
                                fontsize=7, color=col)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.01, step))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(ls="--", lw=0.3, color="0.6")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def plot_polar(fname, theta, curves, title=None, symmetric=True, full=False):
    """Полярна ДС. Якщо symmetric=True, ДС задана на [0, 90°] і дзеркально відображається."""
    fig = plt.figure(figsize=(14 / 2.54, 12 / 2.54))
    ax = fig.add_subplot(111, projection="polar")
    th = np.radians(theta)
    for F, label, style in curves:
        a = np.abs(F)
        ax.plot(th, a, style, lw=1.1, label=label)
        if symmetric:
            ax.plot(-th, a, style, lw=1.1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if not full:
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
    ax.set_thetagrids(range(-90, 91, 15) if not full else range(0, 360, 30))
    ax.set_rlim(0, 1.0)
    ax.set_rticks([0.25, 0.5, 0.707, 1.0])
    ax.set_yticklabels(["0,25", "0,5", "0,707", "1"], fontsize=7)
    ax.grid(ls="--", lw=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(-0.05, -0.05))
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def print_table(title, header, rows):
    """Друк таблиці в консоль."""
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(header)]
    line = "+" + "+".join("-" * x for x in w) + "+"
    print(title)
    print(line)
    print("|" + "|".join(str(h).center(x) for h, x in zip(header, w)) + "|")
    print(line)
    for r in rows:
        print("|" + "|".join(str(c).center(x) for c, x in zip(r, w)) + "|")
    print(line)


# ============================================================================
# Лабораторна робота 4. ДОСЛІДЖЕННЯ ДІЕЛЕКТРИЧНОЇ СТРИЖНЕВОЇ АНТЕНИ (ДСА)
# Одно- та двострижнева антени. Варіант 7
# ============================================================================
import json, os
from scipy.special import j1

lam_list = [3, 3.2, 3.4, 3.5, 3.6, 3.7, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.45, 3, 3, 2.8, 2.8, 3.1, 3.1, 3.6, 3.4, 2.5,
            3.2, 3, 3]
h_list = [5, 8, 10, 12, 14, 15, 16, 15, 14, 12, 10, 8, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
VARIANT = 7
OUT = "figures_lab4"
os.makedirs(OUT, exist_ok=True)

# Параметри стрижня
eps_r, tgd = 2.2, 2e-4
d_max, d_min, d_cp, l = 2.9e-2, 1.7e-2, 2.3e-2, 23.7e-2       # м
lam = lam_list[VARIANT - 1] * 1e-2
h = h_list[VARIANT - 1] * 1e-2
k = 2 * math.pi / lam
print(f"Варіант {VARIANT}: λ = {lam*100:.1f} см, h = {h*100:.0f} см; εr = {eps_r}, dср = {d_cp*100} см, l = {l*100} см")

# Крок 2. Оптимальна довжина хвилі для стрижня з dср (формули (4.9), (4.10))
c_dmax = 1 / math.sqrt(math.pi * (eps_r - 1))          # dmax/λ
c_dmin = 1 / math.sqrt(2.5 * math.pi * (eps_r - 1))    # dmin/λ
lam_opt = d_cp / (0.5 * (c_dmax + c_dmin))
print(f"dmax ≈ {c_dmax:.3f}λ, dmin ≈ {c_dmin:.3f}λ; λопт = 2dср/(dmax/λ + dmin/λ) = {lam_opt*100:.2f} см, f = {3e8/lam_opt/1e9:.2f} ГГц")
print(f"Умова одномодовості (4.1): d ≤ 1,22λ/√(εr−1)... d/λ = {d_cp/lam:.3f}")

# Крок 5. Коефіцієнт уповільнення
xi_opt = 1 + lam / (2 * l)                              # (4.7)
x_graph = d_cp / (2 * lam)                              # аргумент графіка рис. 4.7
v_graph = 0.87                                          # Vф/c для εr ≈ 2,2 при d/(2λ) ≈ 0,3 (рис. 4.7)
xi_graph = 1 / v_graph
xi = xi_opt
print(f"ξопт = 1 + λ/(2l) = {xi_opt:.4f} (Vф/c = {1/xi_opt:.3f}); за рис. 4.7 при d/(2λ) = {x_graph:.3f}: Vф/c ≈ {v_graph}, ξ ≈ {xi_graph:.3f}")
Dm1 = (7 * l / lam, 8 * l / lam)                          # (4.8)
print(f"КСД одного стрижня Dm = (7…8)l/λ = {Dm1[0]:.1f}…{Dm1[1]:.1f}; двох стрижнів ≈ {2*Dm1[0]:.0f}…{2*Dm1[1]:.0f}")
# ККД стрижня (4.11)–(4.13); фактор згасання Rф ≈ 0,68 за рис. 4.8 (εr = 2,2; d/λ ≈ 0,59)
R_f = 0.68
b_S = 1 / l
b_P = math.pi * eps_r * R_f / lam * tgd
eta_A = (b_S + b_P * math.exp(-2 * (b_S + b_P) * l)) / (b_S + b_P)
h_opt = math.sqrt(lam * l)                              # (4.14)
l_opt = lam / (2 * (xi_opt - 1))
print(f"βΣ = {b_S:.3f} 1/м, βП = {b_P:.4f} 1/м, ηA = {eta_A:.4f}; hопт = √(λl) = {h_opt*100:.2f} см, lопт = {l_opt*100:.1f} см")

theta = np.linspace(0, 90, 90001)
t = np.radians(theta)
A = math.pi * l / lam


def F_b(xi_):
    """ДС решітки з біжучою хвилею (4.3), нормована до θ = 0."""
    num = np.sin(A * (xi_ - np.cos(t))) / (xi_ - np.cos(t))
    return np.abs(num) / abs(math.sin(A * (xi_ - 1)) / (xi_ - 1))


def lam1(x):
    x = np.asarray(x, float); out = np.ones_like(x); m = np.abs(x) > 1e-9
    out[m] = 2 * j1(x[m]) / x[m]; return out


Fb = F_b(xi)
F1H = np.abs(lam1(math.pi * d_cp / lam * np.sin(t)))               # (4.5)
F1E = F1H * np.cos(t)                                               # (4.4)
FH1 = F1H * Fb                                                      # одностержнева, площина H
FE1 = F1E * Fb                                                      # одностержнева, площина E
FC = np.abs(np.cos(math.pi * h / lam * np.sin(t)))                  # (4.18)
FH2 = FH1 * FC                                                      # (4.16)
FE2 = FE1 * FC                                                      # (4.17), (4.30)

# Аналітичні «особливі точки» множника Fб (4.19)–(4.21)
nulls_b, max_b = [], []
m = 1
while xi - m * lam / l >= 0:
    nulls_b.append((m, math.degrees(math.acos(min(1, xi - m * lam / l))))); m += 1
m = 1
while xi - (2 * m + 1) * lam / (2 * l) >= 0:
    th_m = math.acos(min(1, xi - (2 * m + 1) * lam / (2 * l)))
    lev = 2 * (l / lam) * (xi - 1) / ((2 * m + 1) * abs(math.sin(A * (xi - 1))))
    max_b.append((m, math.degrees(th_m), lev, lev * math.cos(th_m))); m += 1
hp0 = 61 * math.sqrt(lam / l); z0 = 115 * math.sqrt(lam / l)     # (4.22), (4.23)
# множник двох стрижнів (4.26), (4.27)
nulls_c, max_c = [], []
m = 0
while (2 * m + 1) * lam / (2 * h) <= 1:
    nulls_c.append((m, math.degrees(math.asin((2 * m + 1) * lam / (2 * h))))); m += 1
m = 1
while m * lam / h <= 1:
    max_c.append((m, math.degrees(math.asin(m * lam / h)))); m += 1

print_table("Нулі множника Fб (4.19)", ["m", "θ0, град"], [(a, f"{b:.2f}") for a, b in nulls_b])
print_table("Максимуми бічних пелюсток Fб (4.20), (4.21)", ["m", "θm, град", "FH", "FE=FH·cosθ"],
            [(a, f"{b:.2f}", f"{c:.4f}", f"{d:.4f}") for a, b, c, d in max_b])
print(f"Оцінка ширини (4.22), (4.23): 2θ0 ≈ {z0:.1f}°, 2θ0,5 ≈ {hp0:.1f}°")
print_table("Нулі та максимуми множника FC двох стрижнів", ["m", "θ0C, град", "θmC, град"],
            [(i, f"{nulls_c[i][1]:.2f}" if i < len(nulls_c) else "-", f"{max_c[i-1][1]:.2f}" if 0 < i <= len(max_c) else "-")
             for i in range(max(len(nulls_c), len(max_c) + 1))])

res = {name: analyze(theta, F) for name, F in (("H1", FH1), ("E1", FE1), ("H2", FH2), ("E2", FE2))}
for kk, r in res.items():
    print(kk, f"2θ0,5 = {r['hpbw']:.2f}°, РБП = {r['sll_db']:.1f} дБ, нулі: {[round(x,2) for x in r['nulls'][:6]]}")

# Порівняння з неоптимальним уповільненням (за графіком)
Fb_g = np.sin(A * (xi_graph - np.cos(t))) / (xi_graph - np.cos(t))
Fb_g = np.abs(Fb_g) / np.abs(Fb_g).max()

plot_linear(f"{OUT}/ds_single_linear.png", theta,
            [(Fb, r"$F_б(\theta)$", "C7--"), (FH1, r"$F_H(\theta)$", "C0-"), (FE1, r"$F_E(\theta)$", "C3-")],
            {"H": res["H1"], "E": res["E1"]}, step=10)
plot_polar(f"{OUT}/ds_single_polar.png", theta, [(FH1, r"$F_H(\theta)$", "C0-"), (FE1, r"$F_E(\theta)$", "C3-")])
plot_linear(f"{OUT}/ds_double_linear.png", theta,
            [(FC, r"$F_C(\theta)$", "C7:"), (FH2, r"$F_H(\theta)$", "C0-"), (FE2, r"$F_E(\theta)$", "C3-")],
            {"H": res["H2"], "E": res["E2"]}, step=10, annotate=False)
plot_polar(f"{OUT}/ds_double_polar.png", theta, [(FH2, r"$F_H(\theta)$", "C0-"), (FE2, r"$F_E(\theta)$", "C3-")])
fig, ax = plt.subplots(figsize=(17 / 2.54, 8 / 2.54))
ax.plot(theta, Fb, "C0-", lw=1.2, label=f"ξ = ξопт = {xi_opt:.3f}".replace(".", ","))
ax.plot(theta, Fb_g, "C3--", lw=1.2, label=f"ξ = {xi_graph:.2f} (рис. 4.7, d/(2λ) = {x_graph:.2f})".replace(".", ","))
ax.set_xlim(0, 90); ax.set_ylim(0, 1.05); ax.grid(ls="--", lw=0.3); ax.set_xlabel(r"$\theta$, град")
ax.set_ylabel(r"$F_б(\theta)$ (норм. до max)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/xi_compare.png", dpi=200); plt.close(fig)

json.dump({"variant": VARIANT, "lam": lam, "h": h, "k": k, "eps": eps_r, "tgd": tgd, "dmax": d_max, "dmin": d_min,
           "dcp": d_cp, "l": l, "c_dmax": c_dmax, "c_dmin": c_dmin, "lam_opt": lam_opt, "xi_opt": xi_opt,
           "x_graph": x_graph, "v_graph": v_graph, "xi_graph": xi_graph, "A": A, "Dm1": Dm1, "nulls_b": nulls_b,
           "max_b": max_b, "hp0": hp0, "z0": z0, "nulls_c": nulls_c, "max_c": max_c, "res": res,
           "R_f": R_f, "b_S": b_S, "b_P": b_P, "eta_A": eta_A, "h_opt": h_opt, "l_opt": l_opt, "Fb_g_max_angle": float(theta[np.argmax(np.abs(np.sin(A*(xi_graph-np.cos(t)))/(xi_graph-np.cos(t))))])},
          open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
