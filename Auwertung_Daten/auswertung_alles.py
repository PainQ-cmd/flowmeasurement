# -*- coding: utf-8 -*-
"""
batch_xcorr_analysis_filtered_with_stats_and_flow.py

Enthält alle Daten- und Plot-Schritte aus deinem Originalskript
und ergänzt um einen stark gezoomten Plot zur Visualisierung
der parabolischen Peak-Interpolation im Originalstil.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy import stats
import csv

# --- Datei- und Signalparameter ---
signal1_file = "signal100_1.txt"
signal2_file = "signal100_2.txt"
dt = 4e-9           # Abtastintervall 4 ns
t0 = 0.0            # Startzeitpunkt in s
start_us = 127.0    # Ausschnitt ab 127 µs
stop_us  = 157.0    # bis 157 µs

# --- Geometrie und Material für Volumenstrom ---
L = 0.50                    # Weglänge Sender→Empfänger in m (50 cm)
di = 0.8e-3                 # Innendurchmesser Cannula in m
A = np.pi * (di/2)**2       # Querschnittsfläche in m²
Cphase = 1480.0             # Phasengeschwindigkeit im Fluidmode in m/s

def load_all(path):
    """Lädt alle Zeilen aus einer Datei und wandelt in float-Arrays um."""
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
    # 1) Daten laden
    sig1 = load_all(signal1_file)
    sig2 = load_all(signal2_file)

    # 2) Rohsignal-Plot (erste Messreihe) in mV
    idx_full = 0
    if sig1.shape[0] > idx_full:
        t_us_full   = (np.arange(sig1[idx_full].size) * dt + t0) * 1e6
        raw_full_mV = sig1[idx_full] * 1e3
        plt.figure(figsize=(8, 4))
        plt.plot(t_us_full, raw_full_mV, color='#9467bd', lw=1.5)
        plt.title('Gesamtes Rohsignal mit drei Wellengruppen')
        plt.xlabel('Zeit (µs)')
        plt.ylabel('Amplitude (mV)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('raw_signal_full_mV.png', dpi=300)
        print("Plot 'raw_signal_full_mV.png' gespeichert.")
        plt.show()
    else:
        print("Hinweis: idx_full liegt außerhalb der Messreihen.")

    # 3) Filter-Design
    n   = min(len(sig1), len(sig2))
    fs  = 1.0 / dt
    sos = design_bandpass(fs, lowcut=0.1e6, highcut=5e6, order=4)

    # 4) Speicher für Ergebnisse
    lags_ns     = np.zeros(n)
    flow_ul_min = np.zeros(n)

    # 5) Parameter für Parabel-Demo
    idx_demo  = 0     # welche Messreihe zeigen
    demo_done = False
    zoom_ns   = 20    # ±Zoom-Bereich um den Peak (ns)

    # 6) Verarbeitung aller Messreihen
    for i in range(n):
        # 6.1) Offset entfernen + Filtern
        w1 = sig1[i] - np.mean(sig1[i])
        w2 = sig2[i] - np.mean(sig2[i])
        f1 = sosfiltfilt(sos, w1)
        f2 = sosfiltfilt(sos, w2)

        # 6.2) Zeitfenster auswählen
        t_us = (np.arange(f1.size)*dt + t0) * 1e6
        mask = (t_us >= start_us) & (t_us <= stop_us)
        s1   = f1[mask] - np.mean(f1[mask])
        s2   = f2[mask] - np.mean(f2[mask])

        # 6.3) Kreuzkorrelation & Peak-Interpolation
        corr        = np.correlate(s1, s2, mode='full')
        lag_s       = parabolic_interp(corr, dt)
        lags_ns[i]  = lag_s * 1e9

        # 6.4) Volumenstrom aus Δt
        Vmean           = lag_s * Cphase**2 / (2 * L)
        Q               = A * Vmean
        flow_ul_min[i]  = Q * 1e9 * 60

        # 6.5) Demo: stark gezoomte Parabel-Interpolation im Originalstil
        if i == idx_demo and not demo_done:
            idx_peak = np.argmax(corr)
            if 0 < idx_peak < len(corr)-1:
                # Lag-Achse in ns
                lags_axis_ns = (np.arange(len(corr)) - (len(corr)-1)/2) * dt * 1e9

                # Stützpunkte um den Peak
                y_m1 = corr[idx_peak-1]
                y0   = corr[idx_peak]
                y_p1 = corr[idx_peak+1]

                # Delta und interpolierter Scheitel
                delta     = (y_p1 - y_m1) / (2*(2*y0 - y_p1 - y_m1))
                x_vert_ns = lags_axis_ns[idx_peak] + delta * dt * 1e9

                # Parabel-Koeffizienten
                a  = (y_m1 + y_p1 - 2*y0) / 2
                b  = (y_p1 - y_m1) / 2
                c0 = y0
                y_vert = a*delta**2 + b*delta + c0

                # Feine Parabelkurve
                s_dense    = np.linspace(-1.2, 1.2, 200)
                x_parab_ns = lags_axis_ns[idx_peak] + s_dense * dt * 1e9
                y_parab     = a*s_dense**2 + b*s_dense + c0

                # Plot
                plt.figure(figsize=(6, 4))
                plt.plot(lags_axis_ns, corr,
                         color='#1f77b4', lw=1, alpha=0.6,
                         label='Kreuzkorrelation')
                plt.plot(x_parab_ns, y_parab,
                         color='k', lw=1.5,
                         label='Parabel-Approx.')
                plt.scatter(
                    [lags_axis_ns[idx_peak-1],
                     lags_axis_ns[idx_peak],
                     lags_axis_ns[idx_peak+1]],
                    [y_m1, y0, y_p1],
                    color='crimson', zorder=3,
                    label='Stützpunkte'
                )
                plt.scatter([x_vert_ns], [y_vert],
                            marker='s', color='black', zorder=4,
                            label='Interpolierter Peak')

                # Achsen auf Peak ± zoom_ns setzen
                plt.xlim(x_vert_ns - zoom_ns, x_vert_ns + zoom_ns)

                # Y-Limits eng um den Scheitel
                y_delta  = max(y_parab) - min(y_parab)
                y_margin = 0.1 * y_delta
                plt.ylim(y_vert - y_margin, y_vert + y_margin)

                plt.title(f'Gezoomte Parabel-Interpolation (Messung {i+1})')
                plt.xlabel('Lag (ns)')
                plt.ylabel('Korrelationswert (arb. Einh.)')
                plt.grid(True)
                plt.legend(loc='best', frameon=False)
                plt.tight_layout()
                plt.savefig('parabolic_peak_interpolation_zoomed.png', dpi=300)
                print("Plot 'parabolic_peak_interpolation_zoomed.png' gespeichert.")
                plt.show()
            else:
                print("Hinweis: Peak liegt am Rand, Demo übersprungen.")
            demo_done = True

    # 7) Deskriptive Statistik
    m_dt,   med_dt,   std_dt,   var_dt,   (ci_dt_lo, ci_dt_hi)   = descriptive_stats(lags_ns)
    m_Q,    med_Q,    std_Q,    var_Q,    (ci_Q_lo,  ci_Q_hi )   = descriptive_stats(flow_ul_min)

    # 8) Statistik in Textdatei
    stats_file = "lag_und_flow_statistik.txt"
    with open(stats_file, 'w') as f:
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

    # 9) CSV-Datei für Δt und Q
    csv_file = "lag_und_volumenstrom.csv"
    with open(csv_file, 'w', newline='') as csvf:
        writer = csv.writer(csvf, delimiter=';')
        writer.writerow(["Messung", "Δt (ns)", "Q (µL/min)"])
        for i in range(n):
            writer.writerow([i+1, f"{lags_ns[i]:.3f}", f"{flow_ul_min[i]:.3f}"])
    print(f"Messwerte in '{csv_file}' gespeichert.")

    # 10) Plot: interpolierte Lags über Messungen
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

    # 11) Boxplot der Lags
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        lags_ns,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor='#a6cee3')
    )
    plt.xticks([])
    plt.title("Boxplot der interpolierten Lags bei 100 mBar / 20 mBar")
    plt.ylabel("Lag (ns)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lag_difference_boxplot.png", dpi=300)
    print("Boxplot 'lag_difference_boxplot.png' gespeichert.")
    plt.show()

    # 12) DC-Offset vs. zentriert (Bildpaar)
    idx_plot = 0
    if sig1.shape[0] > idx_plot:
        t_us_full    = (np.arange(sig1[idx_plot].size) * dt + t0) * 1e6
        mask_seg     = (t_us_full >= start_us) & (t_us_full <= stop_us)
        raw_seg      = sig1[idx_plot][mask_seg]
        raw_seg_vis  = raw_seg + (0.5 * np.max(np.abs(raw_seg)))
        raw_seg_vis_mV = raw_seg_vis * 1e3
        cent_seg_mV    = raw_seg_vis_mV - np.mean(raw_seg_vis_mV)
        t_us_seg       = t_us_full[mask_seg]

        if raw_seg.size > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            # mit DC-Offset
            ax1.plot(t_us_seg, raw_seg_vis_mV, color='#1f77b4', lw=1.2)
            ax1.axhline(np.mean(raw_seg_vis_mV),
                        color='r', ls='--', lw=1, label='Mittelwert')
            ax1.fill_between(t_us_seg,
                             np.mean(raw_seg_vis_mV), 0.0,
                             color='gray', alpha=0.2)
            ax1.set_title('Rohsignal mit DC-Offset')
            ax1.set_xlabel('Zeit (µs)')
            ax1.set_ylabel('Amplitude (mV)')
            ax1.legend(loc='upper right', frameon=False)
            # zentriert
            ax2.plot(t_us_seg, cent_seg_mV, color='#2ca02c', lw=1.2)
            ax2.axhline(0.0, color='k', ls='--', lw=1, label='0-Linie')
            ax2.set_title('Zentriertes Signal')
            ax2.set_xlabel('Zeit (µs)')
            ax2.set_ylabel('Amplitude (mV)')
            ax2.legend(loc='upper right', frameon=False)
            plt.tight_layout()
            plt.savefig('dc_offset_bildpaar_mV.png', dpi=300)
            print("Bild 'dc_offset_bildpaar_mV.png' gespeichert.")
            plt.show()
        else:
            print("Hinweis: Keine Datenpunkte im gewählten Zeitfenster.")
    else:
        print("Hinweis: idx_plot außerhalb der Messreihen.")

    # 13) Vergleich ungefiltert vs. gefiltert
    idx_comp = 0
    if sig1.shape[0] > idx_comp:
        raw       = sig1[idx_comp] - np.mean(sig1[idx_comp])
        filt_full = sosfiltfilt(sos, raw)
        t_us_full = (np.arange(raw.size) * dt + t0) * 1e6
        mask      = (t_us_full >= start_us) & (t_us_full <= stop_us)

        raw_seg_mV  = raw[mask]  * 1e3
        filt_seg_mV = filt_full[mask] * 1e3
        t_us_seg    = t_us_full[mask]

        if raw_seg_mV.size > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(t_us_seg, raw_seg_mV,
                     label='Ungefiltert', color='orange', lw=1)
            plt.plot(t_us_seg, filt_seg_mV,
                     label='Gefiltert',   color='blue',   lw=1)
            plt.title('Vergleich: Ungefiltertes vs. gefiltertes Signal')
            plt.xlabel('Zeit (µs)')
            plt.ylabel('Amplitude (mV)')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('filtered_vs_unfiltered_mV.png', dpi=300)
            print("Plot 'filtered_vs_unfiltered_mV.png' gespeichert.")
            plt.show()
        else:
            print("Hinweis: Keine Datenpunkte im Vergleichszeitfenster.")
    else:
        print("Hinweis: idx_comp außerhalb der Messreihen.")

if __name__ == "__main__":
    main()
