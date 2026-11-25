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
# Лабораторна робота 2. ДОСЛІДЖЕННЯ ДИРЕКТОРНОЇ АНТЕНИ
# Варіант 7
# ============================================================================
import json, os

# Вихідні дані (таблиця варіантів)
f_list = [400, 420, 440, 460, 480, 520, 540, 560, 580, 600, 620, 640, 630, 610, 600, 590, 580, 570, 560, 550,
          540, 530, 520, 510, 500]
n_list = [6, 7, 8, 9, 12, 11, 10, 9, 8, 7, 6, 5, 12, 11, 10, 9, 8, 7, 6, 6, 7, 7, 8, 8, 9]
VARIANT = 7
OUT = "figures_lab2"
os.makedirs(OUT, exist_ok=True)

c = 299792458.0
f = f_list[VARIANT - 1] * 1e6      # частота, Гц
N = n_list[VARIANT - 1]            # кількість елементів антени
lam = c / f                        # довжина хвилі, м
d = 0.25 * lam                     # середня відстань між вібраторами, м
k = 2 * math.pi / lam              # хвильове число, рад/м
print(f"Варіант {VARIANT}: f = {f/1e6:.0f} МГц, N = {N}")
print(f"λ = {lam:.4f} м, dср = 0,25λ = {d:.4f} м, k = {k:.4f} рад/м, L = (N-1)dср = {(N-1)*d:.4f} м")

# --- ДС на сітці кутів -------------------------------------------------------
theta = np.linspace(0, 90, 90001)          # крок 0,001°
t = np.radians(theta)
psi = k * d * (1 - np.cos(t))              # (1.15)
with np.errstate(invalid="ignore", divide="ignore"):
    FC = np.where(psi < 1e-12, 1.0, np.abs(np.sin(N * psi / 2) / (N * np.sin(psi / 2))))   # (1.14)
    F1E = np.where(np.abs(np.cos(t)) < 1e-12, 0.0, np.abs(np.cos(np.pi / 2 * np.sin(t)) / np.cos(t)))  # ДС вібратора в пл. E
F1H = np.ones_like(theta)
FH = F1H * FC                                # (1.12)
FE = F1E * FC                                # (1.13)

# --- Аналітичний розрахунок «особливих точок» (п. 1.4.1) ----------------------
nulls_an, maxs_an = [], []
m = 1
while True:                                  # (1.19)
    arg = 1 - m * lam / (N * d)
    if arg < 0:
        break
    nulls_an.append((m, math.degrees(math.acos(arg))))
    m += 1
n = 1
while True:                                  # (1.21), (1.23)
    arg = 1 - (2 * n + 1) * lam / (2 * N * d)
    if arg < -1e-12:
        break
    arg = max(arg, 0.0)
    th_m = math.degrees(math.acos(arg))
    FHm = 1 / (N * abs(math.sin((2 * n + 1) * math.pi / (2 * N))))
    F1Em = math.cos(math.pi / 2 * math.sin(math.radians(th_m))) / math.cos(math.radians(th_m)) if th_m < 89.999 else 0.0
    maxs_an.append((n, th_m, FHm, F1Em, FHm * F1Em))
    n += 1
psi90 = k * d
FH90 = abs(math.sin(N * psi90 / 2) / (N * math.sin(psi90 / 2)))   # відлік при θ = π/2

print_table("Напрями нульового випромінювання, ф-ла (1.19)", ["m", "θ0m, град"],
            [(m, f"{x:.2f}") for m, x in nulls_an])
print_table("Напрями та рівні бічних пелюсток, ф-ли (1.21), (1.23)", ["n", "θmax n, град", "FH", "F1E", "FE"],
            [(n_, f"{a:.2f}", f"{b:.4f}", f"{c_:.4f}", f"{e:.4f}") for n_, a, b, c_, e in maxs_an])
print(f"FH(π/2) = {FH90:.4f}")

# --- Числовий аналіз ДС --------------------------------------------------------
rH = analyze(theta, FH)
rE = analyze(theta, FE)
print(f"2θ0,5 (H) = {rH['hpbw']:.2f}°, 2θ0,5 (E) = {rE['hpbw']:.2f}°")
print("нулі H:", [round(x, 2) for x in rH["nulls"]], " бічні H:", [(round(a, 2), round(b, 3)) for a, b in rH["lobes"]])
print("нулі E:", [round(x, 2) for x in rE["nulls"]], " бічні E:", [(round(a, 2), round(b, 3)) for a, b in rE["lobes"]])

# Наближені оцінки: КСД Dm ≈ 5n (n – кількість директорів = N – 2)
n_dir = N - 2
Dm = 5 * n_dir
print(f"Кількість директорів n = {n_dir}, Dm ≈ 5n = {Dm} ({10*math.log10(Dm):.1f} дБ)")

# --- Система «активний вібратор – рефлектор» (п. 1.5) --------------------------
th2 = np.linspace(0, 180, 18001)
t2 = np.radians(th2)
FH2 = np.abs(np.cos(np.pi / 4 * (1 - np.cos(t2))))                                    # (1.24)
with np.errstate(invalid="ignore", divide="ignore"):
    F1E2 = np.where(np.abs(np.cos(t2)) < 1e-12, 0.0, np.abs(np.cos(np.pi / 2 * np.sin(t2)) / np.cos(t2)))
FE2 = F1E2 * FH2                                                                       # (1.25)
tab2 = [(int(a), FH2[int(a * 100)], FE2[int(a * 100)]) for a in range(0, 181, 30)]
print_table("ДС системи «активний вібратор – рефлектор» (крок π/6)", ["θ, град", "FH", "FE"],
            [(a, f"{b:.3f}", f"{c_:.3f}") for a, b, c_ in tab2])

# --- Графіки ---------------------------------------------------------------------
plot_linear(f"{OUT}/ds_linear.png", theta,
            [(F1E, r"$F_{1E}(\theta)$", "C7--"), (FH, r"$F_H(\theta)=F_C(\theta)$", "C0-"), (FE, r"$F_E(\theta)$", "C3-")],
            {"H": rH, "E": rE})
plot_polar(f"{OUT}/ds_polar.png", theta, [(FH, r"$F_H(\theta)$", "C0-"), (FE, r"$F_E(\theta)$", "C3-")])
fig = plt.figure(figsize=(14 / 2.54, 11 / 2.54))
ax = fig.add_subplot(111, projection="polar")
ax.plot(t2, FH2, "C0-", lw=1.1, label=r"$F_H(\theta)$"); ax.plot(-t2, FH2, "C0-", lw=1.1)
ax.plot(t2, FE2, "C3-", lw=1.1, label=r"$F_E(\theta)$"); ax.plot(-t2, FE2, "C3-", lw=1.1)
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_rlim(0, 1); ax.grid(ls="--", lw=0.3)
ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(-0.05, -0.05))
fig.tight_layout(); fig.savefig(f"{OUT}/ds_reflector.png", dpi=200); plt.close(fig)

json.dump({"variant": VARIANT, "f": f, "N": N, "lam": lam, "d": d, "k": k, "L": (N - 1) * d,
           "nulls_an": nulls_an, "maxs_an": maxs_an, "FH90": FH90, "rH": rH, "rE": rE, "Dm": Dm, "n_dir": n_dir,
           "tab2": [(a, float(b), float(c_)) for a, b, c_ in tab2]},
          open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
