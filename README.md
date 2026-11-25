# Антенні пристрої і комплекси — лабораторні роботи, варіант 7

Кожна лабораторна робота — окремий самодостатній скрипт (окрема гілка репозиторію).

| Гілка (пропозиція) | Файл | Тема |
|---|---|---|
| `lab2-directorna` | `lab2_directorna_antenna.py` | Директорна антена |
| `lab3-horn` | `lab3_horn_antenna.py` | Хвилевідні та рупорні випромінювачі |
| `lab4-dielectric-rod` | `lab4_dielectric_rod_antenna.py` | Діелектрична стрижнева антена |
| `lab5-slot-waveguide` | `lab5_slot_waveguide_antenna.py` | Хвилеводно-щілинна антена |
| `lab7-reflector` | `lab7_reflector_antenna.py` | Дзеркальна (параболічна) антена |

## Запуск

```bash
pip install numpy scipy matplotlib
python lab3_horn_antenna.py
```

Скрипт друкує покроковий розрахунок у консоль і зберігає графіки ДС (лінійні та полярні,
з рівнем 0,707, нулями й бічними пелюстками) та `results.json` у папку `figures_labN/`.
Номер варіанта задається змінною `VARIANT` на початку розрахункової частини.

## Приклад для однієї гілки

```bash
git checkout -b lab3-horn
git add lab3_horn_antenna.py README.md
git commit -m "ЛР3: рупорна антена, варіант 7"
git push -u origin lab3-horn
```
