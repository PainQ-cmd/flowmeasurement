# -*- coding: utf-8 -*-
"""
batch_xcorr_analysis_filtered_with_stats_and_flow.py

Lädt zwei Signal-Dateien mit mehreren Messreihen (je Zeile eine Messung),
entfernt den DC-Anteil, filtert das gesamte Waveform im Band 0.1–5 MHz,
schneidet dann den Bereich von 127–157 µs aus,
berechnet die parabolisch interpolierte Kreuzkorrelations-Laufzeitdifferenz Δt,
führt deskriptive Statistik durch (Mittelwert, Median, Std, Var, 95% CI),
berechnet daraus den Volumenstrom für deionisiertes Wasser (µL/min),
speichert die Statistik in einer Textdatei auf Deutsch im vorgegebenen Format,
speichert zusätzlich alle Messpaare (Δt, Q) in einer CSV-Datei für weitergehende Auswertung,
erstellt die gleichen Plots wie im Originalskript (Kreuzkorrelations-Plot, Boxplot),
speichert die Plots und gibt die Ergebnisse in der Konsole aus.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy import stats
import csv
import os

# --- Datei- und Signalparameter ---
signal1_file = "signal94_1.txt"
signal2_file = "signal94_2.txt"
dt = 4e-9           # Abtastintervall 4 ns
t0 = 0.0            # Startzeitpunkt in s
start_us = 127.0    # Ausschnitt ab 127 µs
stop_us  = 157.0    # bis 157 µs

# --- Geometrie und Material für Volumenstrom ---
L = 0.50                    # Weglänge Sender→Empfänger in m (50 cm)
di = 0.8e-3                 # Innendurchmesser Cannula in m
A = np.pi * (di/2)**2       # Querschnittsfläche in m²
Cphase = 1480.0             # Phasengeschwindigkeit im Fluidmode in m/s

# Setze das Arbeitsverzeichnis auf den Ordner der Skriptdatei
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_all(path):
    """Lädt alle Zeilen aus einer Datei, wandelt in float-Arrays um."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split(",") if x]
            data.append(vals)
    return np.array(data)

def parabolic_interp(corr, dt_s):
    """
    Parabolische Interpolation um den Peak der Kreuzkorrelation
    für sub-sample Zeitauflösung.
    """
    idx = np.argmax(corr)
    lags = (np.arange(len(corr)) - (len(corr)-1)/2) * dt_s
    base = lags[idx]
    if 0 < idx < len(corr)-1:
        y_m1, y0, y_p1 = corr[idx-1], corr[idx], corr[idx+1]
        delta = (y_p1 - y_m1) / (2*(2*y0 - y_p1 - y_m1))
        return base + delta * dt_s
    return base

def design_bandpass(fs, lowcut, highcut, order=4):
    """Erstellt ein Bandpass-Filter (SOS) im Frequenzbereich low–high."""
    nyq = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band', output='sos')

def descriptive_stats(arr):
    """Berechnet Mittelwert, Median, Std, Var und 95% CI des Mittelwerts."""
    n = arr.size
    mean = arr.mean()
    median = np.median(arr)
    std = arr.std(ddof=1)
    var = arr.var(ddof=1)
    sem = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lo = mean - t_crit*sem
    ci_hi = mean + t_crit*sem
    return mean, median, std, var, (ci_lo, ci_hi)

def main():
    # Daten einlesen
    sig1 = load_all(signal1_file)
    sig2 = load_all(signal2_file)
    n = min(len(sig1), len(sig2))

    # Filter entwerfen
    fs = 1.0 / dt
    sos = design_bandpass(fs, lowcut=0.1e6, highcut=5e6, order=4)

    # Arrays für Δt (ns) und Volumenstrom (µL/min)
    lags_ns = np.zeros(n)
    flow_ul_min = np.zeros(n)

    # Schleife über alle Messreihen
    for i in range(n):
        w1 = sig1[i] - np.mean(sig1[i])
        w2 = sig2[i] - np.mean(sig2[i])
        f1 = sosfiltfilt(sos, w1)
        f2 = sosfiltfilt(sos, w2)

        t_us = (np.arange(f1.size)*dt + t0)*1e6
        mask = (t_us >= start_us) & (t_us <= stop_us)
        s1 = f1[mask] - np.mean(f1[mask])
        s2 = f2[mask] - np.mean(f2[mask])

        corr = np.correlate(s1, s2, mode='full')
        lag_s = parabolic_interp(corr, dt)      # in s
        lag_ns = lag_s * 1e9                    # in ns
        lags_ns[i] = lag_ns

        # Volumenstrom Q = A · Vmean, mit Vmean = Δt · Cphase² / (2·L)
        Vmean = lag_s * Cphase**2 / (2 * L)     # m/s
        Q = A * Vmean                           # m³/s
        flow_ul_min[i] = Q * 1e9 * 60           # µL/min

    # Statistik Δt
    m_dt, med_dt, std_dt, var_dt, (ci_dt_lo, ci_dt_hi) = descriptive_stats(lags_ns)
    # Statistik Q
    m_Q, med_Q, std_Q, var_Q, (ci_Q_lo, ci_Q_hi) = descriptive_stats(flow_ul_min)

    # Textdatei mit Statistik (auf Deutsch)
    stats_file = "lag_und_flow_statistik.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Deskriptive Statistik der Laufzeitdifferenzen Δt (ns)\n")
        f.write(f"Anzahl Messwerte   : {n}\n")
        f.write(f"Mittelwert (ns)    : {m_dt:.3f}\n")
        f.write(f"Median (ns)        : {med_dt:.3f}\n")
        f.write(f"Std-Abweichung     : {std_dt:.3f}\n")
        f.write(f"Varianz            : {var_dt:.3f}\n")
        f.write(f"95% CI Mittelwert  : [{ci_dt_lo:.3f}, {ci_dt_hi:.3f}]\n\n")
        f.write("Deskriptive Statistik Volumenstrom Q (µL/min)\n")
        f.write(f"Mittelwert         : {m_Q:.3f}\n")
        f.write(f"Median             : {med_Q:.3f}\n")
        f.write(f"Std-Abweichung     : {std_Q:.3f}\n")
        f.write(f"Varianz            : {var_Q:.3f}\n")
        f.write(f"95% CI Mittelwert  : [{ci_Q_lo:.3f}, {ci_Q_hi:.3f}]\n")
    print(f"Statistik in '{stats_file}' gespeichert.")

    # CSV-Datei für Δt und Q
    csv_file = "lag_und_volumenstrom.csv"
    with open(csv_file, 'w', encoding='utf-8', newline='') as csvf:
        writer = csv.writer(csvf, delimiter=';')
        writer.writerow(["Messung", "Δt (ns)", "Q (µL/min)"])
        for i in range(n):
            writer.writerow([i+1, f"{lags_ns[i]:.3f}", f"{flow_ul_min[i]:.3f}"])
    print(f"Messwerte in '{csv_file}' gespeichert.")

    # Konsole
    print("\n--- Laufzeitdifferenz Δt (ns) ---")
    print(f"Anzahl: {n}")
    print(f"Mittelwert: {m_dt:.3f}, Median: {med_dt:.3f}, Std: {std_dt:.3f}")
    print(f"95% CI: [{ci_dt_lo:.3f}, {ci_dt_hi:.3f}]")
    print("\n--- Volumenstrom Q (µL/min) ---")
    print(f"Mittelwert: {m_Q:.3f}, Median: {med_Q:.3f}, Std: {std_Q:.3f}")
    print(f"95% CI: [{ci_Q_lo:.3f}, {ci_Q_hi:.3f}]\n")

    # Plot 1: Kreuzkorrelations-Lags über Messungen
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n+1), lags_ns, 'o-', linewidth=1)
    plt.title("Interpolierte Lags über alle Messungen")
    plt.xlabel("Messung 100 mBar / 20 mBar")
    plt.ylabel("Lag (ns)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cross_correlation_plot.png", dpi=300)
    print("Plot 'cross_correlation_plot.png' gespeichert.")
    plt.show()

    # --- Checks für lags_ns vor Dichteplot ---
    print("Anzahl Lags:", len(lags_ns))
    print("Min:", np.min(lags_ns), "Max:", np.max(lags_ns))
    print("NaN:", np.isnan(lags_ns).any(), "Inf:", np.isinf(lags_ns).any())
    lags_ns_clean = lags_ns[np.isfinite(lags_ns)]
    # Für den Dichteplot nur die ersten 1000 Werte verwenden
    if len(lags_ns_clean) > 1000:
        print("Warnung: Mehr als 1000 Werte, nur Ausschnitt für Plot!")
        lags_ns_plot = lags_ns_clean[:1000]
    else:
        lags_ns_plot = lags_ns_clean

    # Plot 1b: Dichteplot (KDE) der Lags
    plt.figure(figsize=(8, 4))
    import seaborn as sns
    sns.kdeplot(lags_ns_plot, fill=True, color="tab:blue", alpha=0.5)
    plt.title("Dichteplot der interpolierten Lags")
    plt.xlabel("Lag (ns)")
    plt.ylabel("Wahrscheinlichkeitsdichte (1/ns)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lag_density_plot.png", dpi=300)
    print("Plot 'lag_density_plot.png' gespeichert.")
    plt.show()

    # Plot 2: Boxplot der Lags
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        lags_ns,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor='#a6cee3')
    )
    plt.xticks([])  # X-Tick-Beschriftung entfernen
    plt.title("Boxplot der interpolierten Lags bei 100 mBar / 20 mBar")
    plt.ylabel("Lag (ns)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lag_difference_boxplot.png", dpi=300)
    print("Boxplot 'lag_difference_boxplot.png' gespeichert.")
    plt.show()

if __name__ == "__main__":
    main()
