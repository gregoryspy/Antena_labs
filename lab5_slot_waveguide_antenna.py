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
# Лабораторна робота 5. ДОСЛІДЖЕННЯ ХВИЛЕВОДНО-ЩІЛИННОЇ АНТЕНИ (ХвЩА)
# Варіант 7
# ============================================================================
import json, os

lam_list = [3, 3.2, 3.4, 3.5, 3.65, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.15, 3, 3, 2.8, 2.8, 3.1, 3.1, 3.6, 3.4, 2.5,
            3.2, 3.3, 3.4]
type_list = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
# Таблиця 4: тип антени -> (опис, кількість щілин, довжина щілини, см, відстань між щілинами, см, ζ)
types = {1: ("поздовжні щілини на широкій стінці хвилеводу в «шаховому» порядку", 14, 1.6, 2.0, 1),
         2: ("поздовжні щілини у вузькій стінці хвилеводу", 5, 2.0, 4.0, 0),
         3: ("поперечні щілини у широкій стінці хвилеводу", 6, 1.6, 3.2, 0)}
VARIANT = 7
OUT = "figures_lab5"
os.makedirs(OUT, exist_ok=True)

lam = lam_list[VARIANT - 1] * 1e-2
tp = type_list[VARIANT - 1]
descr, N, l_sl, d, zeta = types[tp]
l_sl *= 1e-2; d *= 1e-2
a_wg = 23e-3                                   # широка стінка хвилеводу МЕК-100, м
lam_cr = 2 * a_wg                              # критична довжина хвилі H10
lam_v = lam / math.sqrt(1 - (lam / lam_cr) ** 2)   # (5.19)
k = 2 * math.pi / lam
dphi = 2 * math.pi * d / lam_v - zeta * math.pi    # фазовий зсув між сусідніми щілинами
print(f"Варіант {VARIANT}: λ = {lam*100:.2f} см, тип {tp}: {descr}; N = {N}, lщ = {l_sl*100} см, d = {d*100} см, ζ = {zeta}")
print(f"k = {k:.2f} рад/м, λкр = 2a = {lam_cr*100:.1f} см, λхв = {lam_v*100:.3f} см, ξ = 2πd/λхв − ζπ = {dphi:.4f} рад ({math.degrees(dphi):.2f}°)")

theta = np.linspace(-90, 90, 180001)
t = np.radians(theta)
if tp in (1, 2):
    psi = k * d * np.sin(t) - 2 * math.pi * d / lam_v + zeta * math.pi     # аргумент (5.21)
else:
    psi = k * d * np.sin(t) - 2 * math.pi * d / lam_v                        # аргумент (5.26)
# множник решітки з урахуванням періодичності 2π
with np.errstate(invalid="ignore", divide="ignore"):
    den = N * np.sin(psi / 2)
    FC = np.abs(np.sin(N * psi / 2) / den)
    FC[np.abs(den) < 1e-9] = 1.0
with np.errstate(invalid="ignore", divide="ignore"):
    F1 = np.abs(np.cos(np.pi / 2 * np.sin(t)) / np.cos(t))              # ДС напівхвильової щілини (5.15)
    F1[np.abs(np.cos(t)) < 1e-12] = 0.0
FH = F1 * FC                                                            # (5.20)
FE = np.ones_like(theta)                                                # (5.16): у площині E щілина ненаправлена

# Напрямок головного максимуму: ψ = 0
s_main = lam / lam_v - zeta * lam / (2 * d)
th_main = math.degrees(math.asin(s_main))
# Нулі (5.22), максимуми (5.23), рівні (5.24)
nulls, maxs = [], []
for m in range(-60, 61):
    if m == 0:
        continue
    s = m * lam / (d * N) + lam / lam_v - zeta * lam / (2 * d)
    if abs(s) <= 1 and (m % N != 0):
        nulls.append((m, math.degrees(math.asin(s))))
for n in range(-60, 61):
    if n in (0, -1):
        continue
    s = (2 * n + 1) * lam / (2 * d * N) + lam / lam_v - zeta * lam / (2 * d)
    if abs(s) <= 1:
        lev = 1 / (N * abs(math.sin((2 * n + 1) * math.pi / (2 * N))))
        tm = math.asin(s)
        f1 = math.cos(math.pi / 2 * math.sin(tm)) / math.cos(tm) if abs(math.cos(tm)) > 1e-9 else 0
        maxs.append((n, math.degrees(tm), lev, f1, lev * f1))
nulls.sort(key=lambda x: x[1]); maxs.sort(key=lambda x: x[1])
z0 = 2 * math.degrees(math.asin(lam / (d * N)))       # (5.25)
# побічні головні максимуми (ψ = ±2π)
grating = [math.degrees(math.asin(s_main + q * lam / d)) for q in (-1, 1) if abs(s_main + q * lam / d) <= 1]
print(f"Головний максимум: sinθгол = λ/λхв − ζλ/(2d) = {s_main:.4f}, θгол = {th_main:.2f}°; дифракційні максимуми: {grating or 'відсутні'}")
print(f"Ширина головної пелюстки за нулями (5.25): 2θ0 = {z0:.2f}°")
print_table("Нулі ДС (5.22)", ["m", "θ0m, град"], [(m, f"{x:.2f}") for m, x in nulls])
print_table("Бічні пелюстки (5.23), (5.24)", ["n", "θmax, град", "FHC", "F1H", "FH"],
            [(n_, f"{a:.2f}", f"{b:.4f}", f"{c:.4f}", f"{e:.4f}") for n_, a, b, c, e in maxs])

rH = analyze(theta, FH)
rC = analyze(theta, FC)
print(f"Числово: θгол = {rH['main']:.2f}°, 2θ0,5 = {rH['hpbw']:.2f}° (від {rH['left']:.2f}° до {rH['right']:.2f}°), РБП = {rH['sll_db']:.1f} дБ")
# КСД лінійної решітки довжиною L = (N−1)d з рівноамплітудним розподілом: D ≈ 2L/λ (оцінка)
L = N * d
D_est = 2 * L / lam
print(f"Довжина антени L = Nd = {L*100:.1f} см, оцінка КСД D ≈ 2L/λ = {D_est:.1f}")

plot_linear(f"{OUT}/ds_H_linear.png", theta,
            [(F1, r"$F_{1H}(\theta)$", "C7--"), (FC, r"$F_{HC}(\theta)$", "C2:"), (FH, r"$F_H(\theta)$", "C0-")],
            {"H": rH}, xlim=(-90, 90), step=15)
# збільшений фрагмент головної пелюстки
plot_linear(f"{OUT}/ds_H_zoom.png", theta,
            [(FH, r"$F_H(\theta)$", "C0-")], {"H": rH}, xlim=(-30, 45), step=5)
plot_linear(f"{OUT}/ds_E_linear.png", theta, [(FE, r"$F_E(\theta)=1$", "C3-")], {}, xlim=(-90, 90), step=15)
plot_polar(f"{OUT}/ds_polar.png", theta, [(FH, r"$F_H(\theta)$", "C0-"), (FE, r"$F_E(\theta)$", "C3--")], symmetric=False)

json.dump({"variant": VARIANT, "lam": lam, "type": tp, "descr": descr, "N": N, "l_sl": l_sl, "d": d, "zeta": zeta,
           "a_wg": a_wg, "lam_cr": lam_cr, "lam_v": lam_v, "k": k, "dphi": dphi, "s_main": s_main, "th_main": th_main,
           "nulls": nulls, "maxs": maxs, "z0": z0, "grating": grating, "rH": rH, "L": L, "D_est": D_est},
          open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
