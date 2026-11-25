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
# Лабораторна робота 3. ДОСЛІДЖЕННЯ ХВИЛЕВОДНИХ ТА РУПОРНИХ ВИПРОМІНЮВАЧІВ
# Розрахунок ДС пірамідального рупора. Варіант 7
# ============================================================================
import json, os

lam_list = [2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3, 3, 2.8, 2.8, 3.1, 3.1, 3.6, 3.4, 2.5,
            3.2, 3, 3]
size_list = [(10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (12, 12), (11, 11), (13, 13), (12, 12), (13, 13),
             (14, 14), (15, 15), (16, 16), (10, 12), (12, 14), (14, 16), (10, 14), (10, 16), (10, 16), (12, 16),
             (14, 16), (10, 14), (12, 16), (14, 16), (10, 16)]
VARIANT = 7
OUT = "figures_lab3"
os.makedirs(OUT, exist_ok=True)

lam = lam_list[VARIANT - 1] * 1e-2          # м
ap, bp = (x * 1e-2 for x in size_list[VARIANT - 1])   # розміри розкриву, м
k = 2 * math.pi / lam
print(f"Варіант {VARIANT}: λ = {lam*100:.1f} см, ap × bp = {ap*100:.0f} × {bp*100:.0f} см")
print(f"k = {k:.2f} рад/м, ap/λ = {ap/lam:.3f}, bp/λ = {bp/lam:.3f}")

theta = np.linspace(0, 90, 90001)
t = np.radians(theta)
uE = np.pi * bp / lam * np.sin(t)
uH = np.pi * ap / lam * np.sin(t)
with np.errstate(invalid="ignore", divide="ignore"):
    FCE = np.where(uE < 1e-12, 1.0, np.sin(uE) / uE)                         # 1-й множник (3.42)
    xH = 2 * ap / lam * np.sin(t)
    FCH = np.cos(uH) / (1 - xH ** 2)                                          # 1-й множник (3.41)
    near = np.abs(1 - xH ** 2) < 1e-6
    FCH[near] = np.pi / 4                                                     # границя 0/0 при x = 1
F1 = (1 + np.cos(t)) / 2                                                      # ДС елемента Гюйгенса
FE = np.abs(FCE) * F1                                                         # (3.42)
FH = np.abs(FCH) * F1                                                         # (3.41)

# --- Площина E (п. 3.7.1) -------------------------------------------------------
E_nulls, E_max = [], []
n = 1
while n * lam / bp <= 1:
    E_nulls.append((n, math.degrees(math.asin(n * lam / bp)))); n += 1
n = 1
while (2 * n + 1) * lam / (2 * bp) <= 1:
    tm = math.asin((2 * n + 1) * lam / (2 * bp))
    lev = 2 / ((2 * n + 1) * math.pi)
    E_max.append((n, math.degrees(tm), lev, (1 + math.cos(tm)) / 2, lev * (1 + math.cos(tm)) / 2)); n += 1
# --- Площина H (п. 3.7.2) -------------------------------------------------------
H_nulls, H_max = [], []
n = 1
while (2 * n + 1) * lam / (2 * ap) <= 1:
    H_nulls.append((n, math.degrees(math.asin((2 * n + 1) * lam / (2 * ap))))); n += 1
n = 1
while (n + 1) * lam / ap <= 1:
    tm = math.asin((n + 1) * lam / ap)
    lev = 1 / abs(1 - 4 * (n + 1) ** 2)
    H_max.append((n, math.degrees(tm), lev, (1 + math.cos(tm)) / 2, lev * (1 + math.cos(tm)) / 2)); n += 1

print_table("Площина E: нулі", ["n", "θ0n, град"], [(a, f"{b:.2f}") for a, b in E_nulls])
print_table("Площина E: бічні пелюстки", ["n", "θmn, град", "FCE", "F1E", "FE"],
            [(a, f"{b:.2f}", f"{c:.4f}", f"{d:.4f}", f"{e:.4f}") for a, b, c, d, e in E_max])
print_table("Площина H: нулі", ["n", "θ0n, град"], [(a, f"{b:.2f}") for a, b in H_nulls])
print_table("Площина H: бічні пелюстки", ["n", "θmn, град", "FCH", "F1H", "FH"],
            [(a, f"{b:.2f}", f"{c:.4f}", f"{d:.4f}", f"{e:.4f}") for a, b, c, d, e in H_max])

rE = analyze(theta, FE)
rH = analyze(theta, FH)
hp_E_est = 53 * lam / bp          # (3.43)
hp_H_est = 80 * lam / ap          # (3.44)
Dm = 4 * math.pi / lam ** 2 * ap * bp * 0.52     # (3.45), (3.46)
RH_opt = ap ** 2 / (3 * lam)      # оптимальні довжини рупора (з умов ψmH = 3π/4, ψmE = π/2)
RE_opt = bp ** 2 / (2 * lam)
print(f"2θ0,5E = {rE['hpbw']:.2f}° (оцінка 53λ/bp = {hp_E_est:.2f}°), 2θ0,5H = {rH['hpbw']:.2f}° (оцінка 80λ/ap = {hp_H_est:.2f}°)")
print(f"КСД Dm = 0,52·4π·ap·bp/λ² = {Dm:.1f} ({10*math.log10(Dm):.2f} дБ); RHопт = {RH_opt*100:.1f} см, REопт = {RE_opt*100:.1f} см")

plot_linear(f"{OUT}/ds_E_linear.png", theta,
            [(FCE, r"$|F_{CE}(\theta)|$", "C7--"), (F1, r"$F_{1E}(\theta)=(1+\cos\theta)/2$", "C2:"), (FE, r"$F_E(\theta)$", "C3-")],
            {"E": rE})
plot_linear(f"{OUT}/ds_H_linear.png", theta,
            [(FCH, r"$|F_{CH}(\theta)|$", "C7--"), (F1, r"$F_{1H}(\theta)=(1+\cos\theta)/2$", "C2:"), (FH, r"$F_H(\theta)$", "C0-")],
            {"H": rH})
plot_polar(f"{OUT}/ds_polar.png", theta, [(FH, r"$F_H(\theta)$", "C0-"), (FE, r"$F_E(\theta)$", "C3-")])

json.dump({"variant": VARIANT, "lam": lam, "ap": ap, "bp": bp, "k": k, "E_nulls": E_nulls, "E_max": E_max,
           "H_nulls": H_nulls, "H_max": H_max, "rE": rE, "rH": rH, "hpE": hp_E_est, "hpH": hp_H_est, "Dm": Dm,
           "RH_opt": RH_opt, "RE_opt": RE_opt}, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
