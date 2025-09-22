# -*- coding: utf-8 -*-
"""
flow_rate_stats_with_pressure_level.py

Lädt Flow-Rate-Daten aus allen Dateien im aktuellen Verzeichnis, die dem Muster
measurement*.txt entsprechen. Für jede Datei werden deskriptive Statistiken
(Standardabweichung, Varianz, 95 % CI) berechnet und in einer Textdatei
gespeichert. Anschließend wird ein Scatter-Plot erzeugt, der die mittleren
Durchflussraten gegen die berechnete Druckstufe (Mittelwert aus den beiden
angegebenen Druckwerten) darstellt.
"""

import os
import glob
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Mapping: measurement-ID → (Druckwert 1, Druckwert 2) in mBar
DRUCK_MAPPING = {
     83: (  0,   0),
     84: ( 50,  20),
     85: ( 20,  50),
     86: ( 60,  20),
     87: ( 20,  70),
     89: ( 20, 100),
     90: ( 20,  20),
     91: ( 70,  20),
     92: ( 20,  80),
     93: ( 80,  20),
     94: (100,  20),
     98: ( 20, 200),
    100: (100,  20),
}

def load_flow_rates(path):
    """
    Liest die Flow-Rate (µl/min) aus der CSV-Datei:
      - Zeile 1 enthält das Start-Timestamp (wird übersprungen)
      - Zeile 2 enthält die Spaltenüberschriften
      - Ab Zeile 3 die Daten: Flow-Rate in der Spalte "Flow Rate (µl/min)"
    Gibt ein numpy-Array aller Float-Werte zurück.
    """
    flow_vals = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # erste Zeile überspringen
        headers = next(reader)
        try:
            idx = headers.index("Flow Rate (µl/min)")
        except ValueError:
            raise ValueError(f"'Flow Rate (µl/min)' nicht in Kopfzeile gefunden: {headers}")

        for row in reader:
            if len(row) <= idx:
                continue
            try:
                flow_vals.append(float(row[idx]))
            except ValueError:
                continue

    if not flow_vals:
        raise ValueError(f"Keine Flow-Daten in Datei '{path}' gefunden.")
    return np.array(flow_vals)

def descriptive_stats(arr):
    """
    Berechnet n, mean, median, std, var und 95 % Konfidenzintervall für arr.
    Für n=1 werden std, var = 0 und CI = [mean, mean].
    """
    n = arr.size
    mean = arr.mean()
    median = np.median(arr)

    if n > 1:
        std = arr.std(ddof=1)
        var = arr.var(ddof=1)
        sem = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_low = mean - t_crit * sem
        ci_high = mean + t_crit * sem
    else:
        std = 0.0
        var = 0.0
        ci_low = ci_high = mean

    return n, mean, median, std, var, ci_low, ci_high

def main():
    # Optional: CLI-Argument für glob-Pattern
    pattern = sys.argv[1] if len(sys.argv) > 1 else "measurement*.txt"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Keine Dateien zum Muster '{pattern}' gefunden.")
        sys.exit(1)

    # Listen für Zusammenfassung
    std_list    = []
    var_list    = []
    ci_lo_list  = []
    ci_hi_list  = []

    # Listen für Scatter-Plot
    scatter_pressures = []
    scatter_flows     = []
    scatter_ids       = []

    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            meas_id = int(base.replace("measurement", ""))
        except ValueError:
            print(f"Überspringe untypischen Dateinamen: {base}")
            continue

        if meas_id not in DRUCK_MAPPING:
            continue

        d1, d2 = DRUCK_MAPPING[meas_id]
        pressure_level = (d1 + d2) / 2.0

        print(f"Verarbeite {base} – Druckstufe: {pressure_level:.1f} mBar")

        try:
            flows = load_flow_rates(path)
        except Exception as e:
            print(f"  Fehler beim Einlesen: {e}")
            continue

        n, mean, median, std, var, ci_lo, ci_hi = descriptive_stats(flows)
        std_list.append(std)
        var_list.append(var)
        ci_lo_list.append(ci_lo)
        ci_hi_list.append(ci_hi)

        # Einzelne Statistikdatei schreiben
        stats_file = f"{base}_statistics.txt"
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write(f"Deskriptive Statistik für {base}\n")
            f.write("Flow-Rate (µl/min)\n")
            f.write(f"Anzahl Messwerte   : {n}\n")
            f.write(f"Mittelwert         : {mean:.3f}\n")
            f.write(f"Median             : {median:.3f}\n")
            f.write(f"Standardabw.       : {std:.3f}\n")
            f.write(f"Varianz            : {var:.3f}\n")
            f.write(f"95 % CI            : [{ci_lo:.3f}, {ci_hi:.3f}]\n")
        print(f"  Statistik gespeichert in '{stats_file}'")

        # Daten für Scatter-Plot sammeln
        scatter_ids.append(meas_id)
        scatter_pressures.append(pressure_level)
        scatter_flows.append(mean)

    # Zusammenfassung aller Statistiken
    if std_list:
        summary_file = "all_measurements_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Zusammenfassung aller Messungen\n\n")
            f.write(f"Dateien-Anzahl        : {len(std_list)}\n")
            f.write(f"Ø Standardabweichung  : {np.mean(std_list):.3f}\n")
            f.write(f"Ø Varianz             : {np.mean(var_list):.3f}\n")
            f.write(f"Ø 95 % CI             : [{np.mean(ci_lo_list):.3f}, {np.mean(ci_hi_list):.3f}]\n")
        print(f"Gesamt-Zusammenfassung in '{summary_file}' gespeichert.")

    # Scatter-Plot: Druckstufe vs. mittlere Durchflussrate
    if scatter_flows:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            scatter_pressures,
            scatter_flows,
            c='tab:blue',
            edgecolor='k',
            s=80
        )
        for pid, x, y in zip(scatter_ids, scatter_pressures, scatter_flows):
            plt.annotate(pid, (x, y), textcoords="offset points", xytext=(0,5), ha='center')

        plt.title("Mittlere Durchflussraten vs. Druckstufe")
        plt.xlabel("Druckstufe (mBar)")
        plt.ylabel("Mittlere Flow-Rate (µl/min)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        scatter_file = "flow_vs_pressure_level_scatter.png"
        plt.savefig(scatter_file, dpi=300)
        plt.close()
        print(f"Scatter-Plot gespeichert in '{scatter_file}'")

    print("Verarbeitung abgeschlossen.")

if __name__ == "__main__":
    main()
