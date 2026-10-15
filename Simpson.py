import os
import re
import sys
from difflib import get_close_matches
import numpy as np
import pandas as pd
from scipy.integrate import simpson

BINS = [
    ("Arom/Alkyl (10-12)",     10.0, 12.0),
    ("Heteroarom (8.5-10)",     8.5, 10.0),
    ("Arom/Phenyl (6.5-8.5)",   6.5,  8.5),
    ("Olefinic (4.5-6.5)",      4.5,  6.5),
    ("O-CH (3.3-4.5)",          3.3,  4.5),
    ("alpha-C (2.0-3.3)",       2.0,  3.3),
    ("Aliphatic (0.5-2.0)",     0.5,  2.0),
    ("Terminal (-0.1-0.5)",    -0.1,  0.5),
]

TMS_WINDOW = (-0.05, 0.05)

SOLVENT_WINDOWS = {
    "CDCl3": [(7.20, 7.30)],
    "DMSO-d6": [(2.45, 2.55), (3.28, 3.38)],
}

OUTPUT_NAME = "Simpson Regional Results.xlsx"

def normalise_solvent(text):
    t = str(text).strip().lower()
    t = (
        t.replace("₃", "3")
         .replace("₆", "6")
         .replace(" ", "")
         .replace("_", "-")
    )

    if t.startswith("cdcl"):
        return "CDCl3"

    if t.startswith("dmso"):
        return "DMSO-d6"

    return None


def key(name):
    return re.sub(r"\s+", "", str(name).strip().lower())

def read_solvent_file(path):
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        xl = pd.read_excel(path, sheet_name=None)
        names = {s.strip().lower(): s for s in xl}

        df = (
            xl[names["solvent"]]
            if "solvent" in names
            else next(iter(xl.values()))
        )
    else:
        df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [str(c).strip() for c in df.columns]

    spec_col = next(
        (
            c for c in df.columns
            if any(word in c.lower()
                   for word in ("spec", "sample", "name"))
        ),
        df.columns[0],
    )

    sol_col = next(
        (c for c in df.columns if "solv" in c.lower()),
        df.columns[1],
    )

    out = {}

    for spec, sol in zip(df[spec_col], df[sol_col]):
        if pd.isna(spec) or pd.isna(sol):
            continue

        norm = normalise_solvent(sol)

        if norm is None:
            print(
                f"[WARNING] Unrecognised solvent '{sol}' "
                f"for '{spec}'. TMS window only."
            )

        out[key(spec)] = norm

    return out

def read_spectrum(path):
    ppm, inten = [], []

    with open(path, "r", encoding="utf-8-sig",
              errors="ignore") as fh:

        for line in fh:
            line = line.strip().strip('"').strip("'").strip()

            if not line:
                continue

            parts = [
                p for p in re.split(
                    r"[\s,;]+", line.replace('"', " ")
                )
                if p
            ]

            if len(parts) < 2:
                continue

            try:
                ppm.append(float(parts[0]))
                inten.append(float(parts[1]))
            except ValueError:
                continue

    if len(ppm) < 2:
        raise ValueError("Fewer than two numeric rows found.")

    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(inten, dtype=float)

    valid = np.isfinite(ppm) & np.isfinite(y)
    ppm, y = ppm[valid], y[valid]

    if len(ppm) < 2:
        raise ValueError("Fewer than two valid numeric rows found.")

    order = np.argsort(ppm)
    ppm, y = ppm[order], y[order]

    unique_ppm, inverse = np.unique(ppm, return_inverse=True)

    if len(unique_ppm) != len(ppm):
        sums = np.bincount(inverse, weights=y)
        counts = np.bincount(inverse)
        y = sums / counts
        ppm = unique_ppm

    if len(ppm) < 2:
        raise ValueError("Fewer than two distinct ppm values found.")

    return ppm, y

def simpson_integral(y, x):
    return float(simpson(y, x=x))


def bin_area(ppm, y, lo, hi):
    lo = max(float(lo), float(ppm.min()))
    hi = min(float(hi), float(ppm.max()))

    if hi <= lo:
        return 0.0

    inside = (ppm > lo) & (ppm < hi)

    x_original = np.concatenate(
        ([lo], ppm[inside], [hi])
    )

    y_original = np.concatenate(
        (
            [np.interp(lo, ppm, y)],
            y[inside],
            [np.interp(hi, ppm, y)],
        )
    )

    dx = np.diff(x_original)
    dx = dx[dx > 0]

    if len(dx) == 0:
        return 0.0

    step = float(np.median(dx))
    interval = hi - lo

    n_intervals = max(2, int(np.ceil(interval / step)))

    if n_intervals % 2 != 0:
        n_intervals += 1

    x_uniform = np.linspace(lo, hi, n_intervals + 1)
    y_uniform = np.interp(x_uniform, x_original, y_original)

    return simpson_integral(y_uniform, x_uniform)


def integrate_bins(ppm, y, windows):
    y = y.copy()

    for lo, hi in windows:
        y[(ppm >= lo) & (ppm <= hi)] = 0.0

    return [
        bin_area(ppm, y, lo, hi)
        for _, lo, hi in BINS
    ]

def to_percent(areas):
    total = sum(areas)

    return [
        100.0 * area / total if total else 0.0
        for area in areas
    ]

def group_four_regions(areas):
    regions = {
        "Aromatic (bins 1-3)": areas[0] + areas[1] + areas[2],
        "Olefinic (bin 4)": areas[3],
        "alpha-Carbonyl (bins 5+6)": areas[4] + areas[5],
        "Aliphatic (bins 7+8)": areas[6] + areas[7],
    }

    total = sum(areas)

    return {
        name: 100.0 * area / total if total else 0.0
        for name, area in regions.items()
    }

def pick_folder():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder = filedialog.askdirectory(
        title="Select folder containing the .txt / .csv spectra"
    )

    root.destroy()
    return folder

def pick_solvent_file():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select solvent information file (Cancel to skip)",
        filetypes=[
            ("Excel or CSV", "*.xlsx *.xlsm *.xls *.csv"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()
    return path or None

def main():

    if len(sys.argv) > 1:
        folder = sys.argv[1]
        solvent_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        folder = pick_folder()
        solvent_path = pick_solvent_file() if folder else None

    if not folder or not os.path.isdir(folder):
        print("No valid folder selected.")
        return

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".txt", ".csv"))
    )

    if solvent_path:
        files = [
            f for f in files
            if os.path.abspath(os.path.join(folder, f))
            != os.path.abspath(solvent_path)
        ]

    if not files:
        print("No .txt or .csv spectra found.")
        return

    solvent_map = {}

    if solvent_path:
        solvent_map = read_solvent_file(solvent_path)
        print(
            f"[OK] Solvent file read "
            f"({len(solvent_map)} entries)."
        )
    else:
        print(
            "[WARNING] No solvent file given. "
            "Only the TMS window will be removed."
        )

    names = [os.path.splitext(f)[0] for f in files]
    spectrum_keys = {key(n) for n in names}

    for solvent_key in solvent_map:
        if solvent_key not in spectrum_keys:
            close = get_close_matches(
                solvent_key,
                list(spectrum_keys),
                n=1,
                cutoff=0.7,
            )

            hint = (
                f" Closest spectrum key: '{close[0]}'."
                if close else ""
            )

            print(
                f"[WARNING] Solvent file lists '{solvent_key}' "
                f"but no spectrum file matches.{hint}"
            )

    rows8, rows4 = [], []

    for f, name in zip(files, names):

        try:
            ppm, y = read_spectrum(os.path.join(folder, f))
        except Exception as e:
            print(f"[ERROR] {f}: {e}")
            continue

        specimen_key = key(name)
        solvent = solvent_map.get(specimen_key)

        if solvent_path and specimen_key not in solvent_map:
            close = get_close_matches(
                specimen_key,
                list(solvent_map),
                n=1,
                cutoff=0.7,
            )

            hint = (
                f" Did you mean '{close[0]}'?"
                if close else ""
            )

            print(
                f"[WARNING] '{name}' not found in solvent file."
                f"{hint} TMS window only."
            )

        windows = [TMS_WINDOW] + SOLVENT_WINDOWS.get(solvent, [])

        areas = integrate_bins(ppm, y, windows)
        pct = to_percent(areas)

        rows8.append({
            "Specimen": name,
            "Solvent": solvent or "not specified",
            **{
                label: percentage
                for (label, _, _), percentage in zip(BINS, pct)
            },
            "Sum %": sum(pct),
        })

        # Four-region summary.
        reg = group_four_regions(areas)

        rows4.append({
            "Specimen": name,
            "Solvent": solvent or "not specified",
            **reg,
            "Sum %": sum(reg.values()),
        })

        print(
            f"[OK] {name} "
            f"({solvent or 'solvent not specified'})"
        )

    if not rows8:
        print("No spectra could be processed successfully.")
        return

    out = os.path.join(folder, OUTPUT_NAME)

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame(rows8).round(2).to_excel(
            xw, sheet_name="8-bin %", index=False
        )

        if rows4:
            pd.DataFrame(rows4).round(2).to_excel(
                xw, sheet_name="4-region %", index=False
            )

    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()