# -*- coding: utf-8 -*-
"""
flow_rate_stats.py

Lädt die Flow-Rate aus der Messdatei (measurement98.txt),
berechnet deskriptive Statistik und speichert Ergebnisse
sowie einen Boxplot.
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Pfad zur Messdatei; falls über Kommandozeile übergeben, sonst "measurement98.txt"
FILENAME = sys.argv[1] if len(sys.argv) > 1 else "measurement94.txt"

def load_flow_rates(path):
    """
    Liest die Flow-Rate (µl/min) aus der CSV-Datei:
    - Zeile 1 enthält das Start-Timestamp (wird übersprungen)
    - Zeile 2 enthält die Spaltenüberschriften
    - Ab Zeile 3 die Daten: Flow-Rate steht in der Spalte "Flow Rate (µl/min)"
    Liefert ein numpy-Array aller geparsten float-Werte.
    """
    flow_vals = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # erste Zeile überspringen
        headers = next(reader)
        try:
            idx_flow = headers.index("Flow Rate (µl/min)")
        except ValueError:
            raise ValueError(f"'Flow Rate (µl/min)' nicht in Kopfzeile gefunden: {headers}")

        for row in reader:
            if len(row) <= idx_flow:
                continue
            try:
                flow_vals.append(float(row[idx_flow]))
            except ValueError:
                # unzulässiger Eintrag überspringen
                continue

    if not flow_vals:
        raise ValueError("Keine Flow-Daten gefunden.")
    return np.array(flow_vals)

def descriptive_stats(arr):
    """
    Berechnet und gibt zurück:
    (n, mean, median, std, var, ci_low, ci_high)
    Für n=1 werden std, var = 0, CI = [mean, mean].
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
    # Daten einlesen
    try:
        flows = load_flow_rates(FILENAME)
    except Exception as e:
        print("Fehler beim Einlesen der Flow-Rate:", e)
        sys.exit(1)

    # Statistik berechnen
    n, mean, median, std, var, ci_lo, ci_hi = descriptive_stats(flows)

    # Statistik in Textdatei speichern
    out_stats = "flow_rate_statistics.txt"
    with open(out_stats, "w", encoding='utf-8') as f:
        f.write("Deskriptive Statistik Flow-Rate (µl/min)\n")
        f.write(f"Anzahl Messwerte   : {n}\n")
        f.write(f"Mittelwert         : {mean:.3f}\n")
        f.write(f"Median             : {median:.3f}\n")
        f.write(f"Standardabweichung : {std:.3f}\n")
        f.write(f"Varianz            : {var:.3f}\n")
        f.write(f"95 % CI            : [{ci_lo:.3f}, {ci_hi:.3f}]\n")
    print(f"Statistik in '{out_stats}' gespeichert.")

    # Konsole-Ausgabe
    print("\n--- Flow-Rate Statistik ---")
    print(f"Anzahl\t: {n}")
    print(f"Mean\t: {mean:.3f}")
    print(f"Median\t: {median:.3f}")
    print(f"Std\t: {std:.3f}")
    print(f"Var\t: {var:.3f}")
    print(f"95 % CI\t: [{ci_lo:.3f}, {ci_hi:.3f}]\n")

    # Boxplot erstellen
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        flows,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="#66c2a5", edgecolor="#238b45")
    )
    plt.title("Boxplot Flow-Rate (µl/min)")
    plt.ylabel("Flow-Rate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Plot speichern
    out_plot = "flow_rate_boxplot.png"
    plt.savefig(out_plot, dpi=300)
    print(f"Boxplot in '{out_plot}' gespeichert.")
    plt.show()

if __name__ == "__main__":
    main()
