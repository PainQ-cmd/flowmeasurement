# -*- coding: utf-8 -*-
"""
flow_rate_stats.py

Lädt Flow-Rate-Daten aus allen Dateien im aktuellen Verzeichnis, die dem Muster
measurement*.txt entsprechen. Für jede Datei werden deskriptive Statistiken
berechnet und in einer Textdatei gespeichert. Zusätzlich wird ein Boxplot für
jeden Datensatz erzeugt und am Ende die Durchschnittswerte der berechneten
Standardabweichungen, Varianzen und 95 % Konfidenzintervalle in einer
Zusammenfassungsdatei abgelegt.
"""

import os
import glob
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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
        next(reader)  # Start-Timestamp überspringen
        headers = next(reader)
        try:
            idx_flow = headers.index("Flow Rate (µl/min)")
        except ValueError:
            raise ValueError(
                f"'Flow Rate (µl/min)' nicht in Kopfzeile gefunden: {headers}"
            )

        for row in reader:
            if len(row) <= idx_flow:
                continue
            try:
                flow_vals.append(float(row[idx_flow]))
            except ValueError:
                continue  # ungültige Einträge überspringen

    if not flow_vals:
        raise ValueError(f"Keine Flow-Daten in Datei '{path}' gefunden.")
    return np.array(flow_vals)


def descriptive_stats(arr):
    """
    Berechnet und gibt zurück:
      (n, mean, median, std, var, ci_low, ci_high)
    Für n=1 werden std, var = 0 und CI = [mean, mean].
    """
    n = arr.size
    mean = arr.mean()
    median = np.median(arr)

    if n > 1:
        std = arr.std(ddof=1)
        var = arr.var(ddof=1)
        sem = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_low = mean - t_crit * sem
        ci_high = mean + t_crit * sem
    else:
        std = 0.0
        var = 0.0
        ci_low = ci_high = mean

    return n, mean, median, std, var, ci_low, ci_high


def main():
    # Muster für Eingabedateien: measurement*.txt oder über CLI
    pattern = sys.argv[1] if len(sys.argv) > 1 else "measurement*.txt"
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        print(f"Keine Dateien gefunden, die zum Muster '{pattern}' passen.")
        sys.exit(1)

    # Liste zur Aggregation aller berechneten Zahlen
    std_list = []
    var_list = []
    ci_lo_list = []
    ci_hi_list = []

    for path in file_list:
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"\nVerarbeite Datei: {path}")

        # Daten einlesen
        try:
            flows = load_flow_rates(path)
        except Exception as e:
            print(f"  Fehler beim Einlesen der Datei '{path}': {e}")
            continue

        # Statistik berechnen
        n, mean, median, std, var, ci_lo, ci_hi = descriptive_stats(flows)

        # Werte zur späteren Durchschnittsbildung sammeln
        std_list.append(std)
        var_list.append(var)
        ci_lo_list.append(ci_lo)
        ci_hi_list.append(ci_hi)

        # Statistik in Textdatei speichern
        out_stats = f"{base}_statistics.txt"
        with open(out_stats, "w", encoding='utf-8') as f:
            f.write(f"Deskriptive Statistik für {base}\n")
            f.write("Flow-Rate (µl/min)\n")
            f.write(f"Anzahl Messwerte   : {n}\n")
            f.write(f"Mittelwert         : {mean:.3f}\n")
            f.write(f"Median             : {median:.3f}\n")
            f.write(f"Standardabweichung : {std:.3f}\n")
            f.write(f"Varianz            : {var:.3f}\n")
            f.write(f"95 % CI            : [{ci_lo:.3f}, {ci_hi:.3f}]\n")
        print(f"  Statistik in '{out_stats}' gespeichert.")

        # Konsolenausgabe
        print("  --- Flow-Rate Statistik ---")
        print(f"  Anzahl : {n}")
        print(f"  Mean   : {mean:.3f}")
        print(f"  Median : {median:.3f}")
        print(f"  Std    : {std:.3f}")
        print(f"  Var    : {var:.3f}")
        print(f"  95 % CI: [{ci_lo:.3f}, {ci_hi:.3f}]\n")

        # Boxplot erstellen und speichern
        plt.figure(figsize=(6, 4))
        plt.boxplot(
            flows,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#66c2a5", edgecolor="#238b45")
        )
        plt.title(f"Boxplot Flow-Rate (µl/min): {base}")
        plt.ylabel("Flow-Rate")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_plot = f"{base}_boxplot.png"
        plt.savefig(out_plot, dpi=300)
        plt.close()
        print(f"  Boxplot in '{out_plot}' gespeichert.")

    # Durchschnittswerte berechnen und in Zusammenfassung schreiben
    if std_list:
        avg_std    = np.mean(std_list)
        avg_var    = np.mean(var_list)
        avg_ci_lo  = np.mean(ci_lo_list)
        avg_ci_hi  = np.mean(ci_hi_list)
        count      = len(std_list)

        out_summary = "all_measurements_summary.txt"
        with open(out_summary, "w", encoding='utf-8') as f:
            f.write(
                "Durchschnitt der berechneten Standardabweichungen, "
                "Varianzen und 95 % CI aller Messungen\n"
            )
            f.write(f"Anzahl Dateien                      : {count}\n")
            f.write(f"Durchschn. Standardabweichung (σ̄)  : {avg_std:.3f}\n")
            f.write(f"Durchschn. Varianz (σ²̄)            : {avg_var:.3f}\n")
            f.write(
                f"Durchschn. 95 % CI                  : "
                f"[{avg_ci_lo:.3f}, {avg_ci_hi:.3f}]\n"
            )

        print(f"\nZusammenfassung in '{out_summary}' gespeichert.")

    print("\nVerarbeitung abgeschlossen.")


if __name__ == "__main__":
    main()
