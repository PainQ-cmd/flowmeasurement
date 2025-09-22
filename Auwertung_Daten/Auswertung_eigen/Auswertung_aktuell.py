# -*- coding: utf-8 -*-
"""
export_and_plot_id94_with_coeff_interp.py

Skript für Mess-ID 94:
- Speichert das gesamte Rohsignal (Kanal 1, erster Sweep) un- und gefiltert als TXT.
- Erstellt hochauflösende Plots:
    * Rohsignal gesamt ungefiltert
    * Rohsignal gesamt gefiltert
    * Dritte Wellengruppe gefiltert
    * Kreuzkorrelations-Koeffizient und parabolische Interpolation am Peak
  jeweils als PNG und PDF.
- Berechnet anschließend wie gewohnt Laufzeitdifferenzen und Volumenstrom.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy import stats

# --- Parameter ---
dt          = 4e-9              # Abtastintervall [s]
start_us    = 127.0             # Segment-Anfang [µs]
stop_us     = 157.0             # Segment-Ende [µs]

L           = 0.50              # Weglänge Sender→Empfänger [m]
di          = 0.8e-3            # Innendurchmesser Cannula [m]
A           = np.pi*(di/2)**2   # Querschnittsfläche [m²]
Cphase      = 1480.0            # Phasengeschwindigkeit [m/s]

FLOW_OFFSET = -149.024          # Offset [µL/min]

# Auswahl der Mess-ID
TARGET_ID = 98

def load_all(path):
    """Lädt alle Zeilen aus einer Datei und gibt sie als 2D-array zurück."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split(",") if x]
            data.append(vals)
    return np.array(data)

def design_bandpass(fs, lowcut, highcut, order=4):
    """Entwirft ein digitales Butterworth-Bandpassfilter (SOS)."""
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')

def parabolic_interp(corr, dt_s):
    """Parabolische Interpolation um den Peak der Kreuzkorrelation."""
    idx  = np.argmax(corr)
    lags = (np.arange(len(corr)) - (len(corr)-1)/2) * dt_s
    base = lags[idx]
    if 0 < idx < len(corr)-1:
        y_m1, y0, y_p1 = corr[idx-1], corr[idx], corr[idx+1]
        delta = (y_p1 - y_m1) / (2*(2*y0 - y_p1 - y_m1))
        return base + delta*dt_s
    return base

def descriptive_stats(arr):
    """Berechnet Mittelwert, Median, Std, Var und 95 % CI des Mittelwerts."""
    n      = arr.size
    mean   = arr.mean()
    median = np.median(arr)
    if n > 1:
        std   = arr.std(ddof=1)
        sem   = std / np.sqrt(n)
        t_val = stats.t.ppf(0.975, df=n-1)
        ci_lo = mean - t_val*sem
        ci_hi = mean + t_val*sem
    else:
        std = ci_lo = ci_hi = 0.0
    return mean, median, std, arr.var(ddof=1), ci_lo, ci_hi

def main():
    fs  = 1.0 / dt
    sos = design_bandpass(fs, lowcut=0.1e6, highcut=5e6, order=4)

    fn1 = f"signal{TARGET_ID}_1.txt"
    fn2 = f"signal{TARGET_ID}_2.txt"
    if not (os.path.exists(fn1) and os.path.exists(fn2)):
        print(f"Dateien für ID {TARGET_ID} fehlen.")
        return

    sig1 = load_all(fn1)
    sig2 = load_all(fn2)
    n    = min(len(sig1), len(sig2))

    # --- 1) Rohsignal gesamt ungefiltert ---
    raw_wave = sig1[0]
    t_us     = np.arange(raw_wave.size) * dt * 1e6
    amp_mV   = raw_wave * 1e3

    # Text-Ausgabe
    raw_fn = f"signal{TARGET_ID}_raw_unfiltered.txt"
    with open(raw_fn, 'w', encoding='utf-8') as g:
        g.write("# Zeit_us,Amplitude [mV]\n")
        for t, a in zip(t_us, amp_mV):
            g.write(f"{t:.6f},{a:.6f}\n")

    # Plot und Speichern
    plt.figure(figsize=(10, 4))
    plt.plot(t_us, amp_mV, color='black', linewidth=0.8)
    plt.xlabel("Zeit [µs]", fontsize=12)
    plt.ylabel("Amplitude [mV]", fontsize=12)
    plt.title("Rohsignal gesamt ungefiltert", fontsize=14)
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    png1 = f"signal{TARGET_ID}_raw_unfiltered.png"
    pdf1 = f"signal{TARGET_ID}_raw_unfiltered.pdf"
    plt.savefig(png1, dpi=600)
    plt.savefig(pdf1, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Ungefiltertes Rohsignal in '{png1}' und '{pdf1}' gespeichert.")

    # --- 2) Rohsignal gesamt gefiltert ---
    w_raw    = raw_wave - raw_wave.mean()
    filt_raw = sosfiltfilt(sos, w_raw)
    amp_filt = filt_raw * 1e3

    filt_raw_fn = f"signal{TARGET_ID}_raw_filtered.txt"
    with open(filt_raw_fn, 'w', encoding='utf-8') as g:
        g.write("# Zeit_us,Amplitude [mV]\n")
        for t, a in zip(t_us, amp_filt):
            g.write(f"{t:.6f},{a:.6f}\n")

    plt.figure(figsize=(10, 4))
    plt.plot(t_us, amp_filt, color='tab:blue', linewidth=0.8)
    plt.xlabel("Zeit [µs]", fontsize=12)
    plt.ylabel("Amplitude [mV]", fontsize=12)
    plt.title("Rohsignal gesamt gefiltert", fontsize=14)
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    png2 = f"signal{TARGET_ID}_raw_filtered.png"
    pdf2 = f"signal{TARGET_ID}_raw_filtered.pdf"
    plt.savefig(png2, dpi=600)
    plt.savefig(pdf2, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Gefiltertes Rohsignal in '{png2}' und '{pdf2}' gespeichert.")

    # --- 3) Dritte Wellengruppe gefiltert ---
    sweep = sig1[0] - sig1[0].mean()
    filt  = sosfiltfilt(sos, sweep)
    t_us  = np.arange(filt.size) * dt * 1e6
    mask  = (t_us >= start_us) & (t_us <= stop_us)
    seg_t = t_us[mask]
    seg_a = filt[mask] * 1e3

    seg_fn = f"signal{TARGET_ID}_third_group.txt"
    with open(seg_fn, 'w', encoding='utf-8') as g:
        g.write("# Zeit_us,Amplitude [mV]\n")
        for t, a in zip(seg_t, seg_a):
            g.write(f"{t:.6f},{a:.6f}\n")

    plt.figure(figsize=(8, 4))
    plt.plot(seg_t, seg_a, color='tab:green', linewidth=1.0)
    plt.xlabel("Zeit [µs]", fontsize=12)
    plt.ylabel("Amplitude [mV]", fontsize=12)
    plt.title("Dritte Wellengruppe gefiltert", fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    png3 = f"signal{TARGET_ID}_third_group.png"
    pdf3 = f"signal{TARGET_ID}_third_group.pdf"
    plt.savefig(png3, dpi=600)
    plt.savefig(pdf3, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Dritte Wellengruppe in '{png3}' und '{pdf3}' gespeichert.")

    # --- 4) Korrelationskoeffizient & Parabel-Interpolation ---
    sweep1 = sig1[0] - sig1[0].mean()
    sweep2 = sig2[0] - sig2[0].mean()
    filt1  = sosfiltfilt(sos, sweep1)
    filt2  = sosfiltfilt(sos, sweep2)

    t_full    = np.arange(filt1.size) * dt * 1e6
    mask_full = (t_full >= start_us) & (t_full <= stop_us)
    s1_seg    = filt1[mask_full] - np.mean(filt1[mask_full])
    s2_seg    = filt2[mask_full] - np.mean(filt2[mask_full])

    s1_norm   = s1_seg / np.linalg.norm(s1_seg)
    s2_norm   = s2_seg / np.linalg.norm(s2_seg)
    corr_coef = np.correlate(s1_norm, s2_norm, mode='full')
    lags_ns   = (np.arange(len(corr_coef)) - (len(corr_coef)-1)/2) * dt * 1e9

    idx       = np.argmax(corr_coef)
    start_idx = max(0, idx-5)
    end_idx   = min(len(corr_coef), idx+6)
    region    = slice(start_idx, end_idx)

    x_vals = lags_ns[idx-1:idx+2]
    y_vals = corr_coef[idx-1:idx+2]
    p_coef = np.polyfit(x_vals, y_vals, 2)
    x_fit  = np.linspace(x_vals[0], x_vals[2], 200)
    y_fit  = np.polyval(p_coef, x_fit)

    x_peak = -p_coef[1] / (2 * p_coef[0])
    y_peak = np.polyval(p_coef, x_peak)

    plt.figure(figsize=(8, 4))
    plt.scatter(lags_ns[region], corr_coef[region],
                color='black', s=20, label='Korrelationskoeffizient')
    plt.plot(x_fit, y_fit, color='tab:blue', linewidth=1.2,
             label='Parabelfit')
    plt.scatter(x_peak, y_peak, color='tab:orange',
                s=50, marker='x', label='Interpolierter Peak')

    plt.xlabel("Lag [ns]", fontsize=12)
    plt.ylabel("Korrelationskoeffizient [–]", fontsize=12)
    plt.title("Kreuzkorrelations-Koeffizient und Interpolation", fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()

    png4 = f"signal{TARGET_ID}_corrcoef_interp.png"
    pdf4 = f"signal{TARGET_ID}_corrcoef_interp.pdf"
    plt.savefig(png4, dpi=600)
    plt.savefig(pdf4, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Korrelations-Plot in '{png4}' und '{pdf4}' gespeichert.")

    # --- Restliche Auswertung (Kreuzkorrelation & Durchfluss) unverändert ---
    lags_ns_arr  = np.zeros(n)
    flows_ul_arr = np.zeros(n)
    for i in range(n):
        w1    = sig1[i] - sig1[i].mean()
        w2    = sig2[i] - sig2[i].mean()
        f1f   = sosfiltfilt(sos, w1)
        f2f   = sosfiltfilt(sos, w2)

        t_us_f   = np.arange(f1f.size) * dt * 1e6
        mask_f   = (t_us_f >= start_us) & (t_us_f <= stop_us)
        seg1     = f1f[mask_f] - np.mean(f1f[mask_f])
        seg2     = f2f[mask_f] - np.mean(f2f[mask_f])

        corr_f   = np.correlate(seg1, seg2, mode='full')
        lag_s    = parabolic_interp(corr_f, dt)
        lags_ns_arr[i]  = lag_s * 1e9

        Vmean    = lag_s * Cphase**2 / (2 * L)
        flows_ul_arr[i] = A * Vmean * 1e9 * 60  # µL/min

    flows_ul_corr = flows_ul_arr - FLOW_OFFSET
    mean_l, med_l, std_l, var_l, ci_lo_l, ci_hi_l = descriptive_stats(lags_ns_arr)
    mean_f, med_f, std_f, var_f, ci_lo_f, ci_hi_f = descriptive_stats(flows_ul_corr)

    stats_fn = f"signal{TARGET_ID}_statistics.txt"
    with open(stats_fn, 'w', encoding='utf-8') as f:
        f.write(f"Statistik ID {TARGET_ID}\n\n")
        f.write("Laufzeitdifferenzen Δt (ns)\n")
        f.write(f"  Anzahl Messwerte   : {n}\n")
        f.write(f"  Mittelwert         : {mean_l:.3f}\n")
        f.write(f"  Median             : {med_l:.3f}\n")
        f.write(f"  Std-Abweichung     : {std_l:.3f}\n")
        f.write(f"  Varianz            : {var_l:.3f}\n")
        f.write(f"  95 % CI Mittelwert : [{ci_lo_l:.3f}, {ci_hi_l:.3f}]\n\n")
        f.write("Korrigierter Volumenstrom Q (µL/min)\n")
        f.write(f"  Mittelwert         : {mean_f:.3f}\n")
        f.write(f"  Median             : {med_f:.3f}\n")
        f.write(f"  Std-Abweichung     : {std_f:.3f}\n")
        f.write(f"  Varianz            : {var_f:.3f}\n")
        f.write(f"  95 % CI Mittelwert : [{ci_lo_f:.3f}, {ci_hi_f:.3f}]\n")
    print(f"Statistik in '{stats_fn}' gespeichert.")

    print("Verarbeitung abgeschlossen.")

if __name__ == "__main__":
    main()
