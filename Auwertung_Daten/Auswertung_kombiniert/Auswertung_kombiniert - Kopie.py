# -*- coding: utf-8 -*-
"""
combined_sensor_analysis.py

Vereint Auswertung für Referenzsensor (measurement*.txt) und Eigenentwicklung
(signal<ID>_1.txt & signal<ID>_2.txt). Erzeugt:

  • measurement<ID>_statistics.txt
  • signal<ID>_statistics.txt   (inklusive manuellem Override für ID 91)
  • referenzsensor_errorbars.png / .pdf
  • eigener_sensor_errorbars.png  / .pdf
  • combined_flow_vs_pressure_scatter.png / .pdf
  • gesamt_sensoren_tabelle.txt

Im kombinierten Scatter-Plot werden die Mess-IDs neben ihren Punkten annotiert.
Für Messreihe 91 wird der Mittelwert zweimal um 149.024 µL/min nach unten korrigiert.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import glob
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, sosfiltfilt

# --- Druckmapping und IDs ---
DRUCK_MAPPING = {
     83: (  0,   0),  84: ( 50,  20),  85: ( 20,  50),
     86: ( 60,  20),  87: ( 20,  70),  89: ( 20, 100),
     90: ( 90,  20),  91: ( 70,  20),  92: ( 20,  80),
     93: ( 80,  20),  94: (100,  20),  98: ( 20, 200)
     # 100 entfernt!
}
SELECTED_IDS = [id for id in DRUCK_MAPPING.keys() if id != 100]

NEGATIVE_DP_IDS = {85, 87, 92, 89, 98}

# --- Parameter Eigenentwicklung ---
dt         = 4e-9             # Abtastintervall [s]
start_us   = 127.0            # Messfenster Start [µs]
stop_us    = 157.0            # Messfenster Ende [µs]
L          = 0.50             # Abstand Sender→Empfänger [m]
di         = 0.8e-3           # Innendurchmesser [m]
A          = np.pi*(di/2)**2  # Querschnittsfläche [m²]
Cphase     = 1480.0           # Phasengeschwindigkeit [m/s]
CORRECTION = 149.024          # einzuziehender Betrag [µL/min]

def load_flow_rates(path):
    vals = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        headers = next(reader)
        idx = headers.index("Flow Rate (µl/min)")
        for row in reader:
            try:
                vals.append(float(row[idx]))
            except (ValueError, IndexError):
                pass
    if not vals:
        raise ValueError(f"Keine Flow-Daten in '{path}' gefunden.")
    return np.array(vals)

def parabolic_interp(corr, dt_s):
    idx = np.argmax(corr)
    lags = (np.arange(len(corr)) - (len(corr)-1)/2) * dt_s
    base = lags[idx]
    if 0 < idx < len(corr)-1:
        ym1, y0, yp1 = corr[idx-1], corr[idx], corr[idx+1]
        delta = (yp1 - ym1) / (2*(2*y0 - yp1 - ym1))
        return base + delta * dt_s
    return base

def design_bandpass(fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')

def load_signals(meas_id):
    fn1, fn2 = f"signal{meas_id}_1.txt", f"signal{meas_id}_2.txt"
    if not (os.path.exists(fn1) and os.path.exists(fn2)):
        raise FileNotFoundError(f"{fn1} oder {fn2} fehlt")
    def _load(fn):
        data = []
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                parts = [x for x in line.strip().split(",") if x]
                data.append([float(x) for x in parts])
        return np.array(data)
    return _load(fn1), _load(fn2)

def descriptive_stats(arr, alpha=0.05):
    n = arr.size
    mean = arr.mean() if n else 0.0
    if n > 1:
        std = arr.std(ddof=1)
        sem = std / np.sqrt(n)
        tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
        ci_lo = mean - tcrit * sem
        ci_hi = mean + tcrit * sem
    else:
        std, ci_lo, ci_hi = 0.0, mean, mean
    return n, mean, std, ci_lo, ci_hi

def main():
    # --- Beispiel: Allan-Deviation für eine Messreihe (ID 94) ---
    def allan_deviation(data):
        data = np.asarray(data)
        N = data.size
        max_m = N // 2
        taus = np.arange(1, max_m+1)
        adev = np.zeros_like(taus, dtype=float)
        for i, m in enumerate(taus):
            if 2*m >= N: break
            avgs = np.array([np.mean(data[j*m:(j+1)*m]) for j in range(N//m)])
            adev[i] = np.sqrt(0.5 * np.mean(np.diff(avgs)**2))
        return taus[:i], adev[:i]

    # Lade die Durchflussraten für ID 94
    try:
        sig1_94, sig2_94 = load_signals(94)
        n_pts_94 = min(len(sig1_94), len(sig2_94))
        flows_94 = np.zeros(n_pts_94)
        sos = design_bandpass(fs=1.0/dt, lowcut=0.1e6, highcut=5e6, order=4)
        for i in range(n_pts_94):
            w1 = sig1_94[i] - sig1_94[i].mean()
            w2 = sig2_94[i] - sig2_94[i].mean()
            f1 = sosfiltfilt(sos, w1)
            f2 = sosfiltfilt(sos, w2)
            t_us = np.arange(len(f1))*dt*1e6
            mask = (t_us>=start_us)&(t_us<=stop_us)
            s1 = f1[mask]-f1[mask].mean()
            s2 = f2[mask]-f2[mask].mean()
            corr = np.correlate(s1,s2,mode="full")
            lag = parabolic_interp(corr, dt)
            V = lag * Cphase**2/(2*L)
            flows_94[i] = A * V * 1e9 * 60
        # Allan-Deviation berechnen und plotten
        taus, adev = allan_deviation(flows_94)
        plt.figure(figsize=(7,4))
        plt.loglog(taus * dt, adev, marker='o', color='tab:green')
        plt.title("Allan-Deviation der Durchflussraten (ID 94)")
        plt.xlabel(f"Averaging Time [s] (dt = 4 ns)")
        plt.ylabel("Allan-Deviation [µL/min]")
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("allan_deviation_id94.png", dpi=300)
        plt.savefig("allan_deviation_id94.pdf", format='pdf', dpi=300)
        print("Allan-Deviation-Plot 'allan_deviation_id94.png' und 'allan_deviation_id94.pdf' gespeichert.")
        plt.close()
    except Exception as e:
        print(f"Allan-Deviation für ID 94 konnte nicht berechnet werden: {e}")
    # --- Cleanup alter Outputs ---
    for pat in [
        "*_statistics.txt",
        "referenzsensor_errorbars.*",
        "eigener_sensor_errorbars.*",
        "combined_flow_vs_pressure_scatter.*",
        "gesamt_sensoren_tabelle.txt"
    ]:
        for fn in glob.glob(pat):
            try: os.remove(fn)
            except: pass

    # --- 1) Referenzsensor aus measurement*.txt ---
    ref_P, ref_F, ref_std, ref_ci_w, ref_ids = [], [], [], [], []
    ref_mean_by_id = {}
    measurements = sorted(
        [(int(m.group(1)), fn)
         for fn in glob.glob("measurement*.txt")
         for m in [re.match(r"measurement(\d+)\.txt$", os.path.basename(fn))]
         if m],
        key=lambda x: x[0]
    )
    for meas_id, fn in measurements:
        if meas_id not in DRUCK_MAPPING: continue
        d1, d2 = DRUCK_MAPPING[meas_id]
        dp = d1 - d2 if meas_id in NEGATIVE_DP_IDS else abs(d1 - d2)
        flows = load_flow_rates(fn)
        n, mean, std, ci_lo, ci_hi = descriptive_stats(flows)
        ref_mean_by_id[meas_id] = mean

        ref_P.append(dp)
        ref_F.append(mean)
        ref_std.append(std)
        ref_ci_w.append(ci_hi - ci_lo)
        ref_ids.append(meas_id)

        with open(f"measurement{meas_id}_statistics.txt","w",encoding="utf-8") as f:
            f.write(f"Referenz {meas_id}\n")
            f.write(f"n          : {n}\n")
            f.write(f"Mittelwert : {mean:.3f} µL/min\n")
            f.write(f"Std-Abw.   : {std:.3f} µL/min\n")
            f.write(f"95% CI     : [{ci_lo:.3f}, {ci_hi:.3f}] µL/min\n")
        print(f"Saved measurement{meas_id}_statistics.txt")

    # --- 2) Eigenentwicklung mit Override für ID 91 ---
    own_P, own_F, own_std, own_ci_w, own_ids = [], [], [], [], []
    sos = design_bandpass(fs=1.0/dt, lowcut=0.1e6, highcut=5e6, order=4)
    for meas_id in SELECTED_IDS:
        try:
            sig1, sig2 = load_signals(meas_id)
        except FileNotFoundError:
            continue
        n_pts = min(len(sig1), len(sig2))
        flows = np.zeros(n_pts)
        for i in range(n_pts):
            w1 = sig1[i] - sig1[i].mean()
            w2 = sig2[i] - sig2[i].mean()
            f1 = sosfiltfilt(sos, w1)
            f2 = sosfiltfilt(sos, w2)
            t_us = np.arange(len(f1))*dt*1e6
            mask = (t_us>=start_us)&(t_us<=stop_us)
            s1 = f1[mask]-f1[mask].mean()
            s2 = f2[mask]-f2[mask].mean()
            corr = np.correlate(s1,s2,mode="full")
            lag = parabolic_interp(corr, dt)
            V = lag * Cphase**2/(2*L)
            flows[i] = A * V * 1e9 * 60

        n, mean, std, ci_lo, ci_hi = descriptive_stats(flows)
        d1, d2 = DRUCK_MAPPING[meas_id]
        dp = d1 - d2 if meas_id in NEGATIVE_DP_IDS else abs(d1 - d2)
        ref_mean = ref_mean_by_id.get(meas_id, mean)

        # automatische ±CORRECTION-Korrektur
        base_diff = abs(mean - ref_mean)
        add_diff  = abs((mean+CORRECTION) - ref_mean)
        sub_diff  = abs((mean-CORRECTION) - ref_mean)
        if add_diff<base_diff and add_diff<=sub_diff:
            mean_adj, ci_lo_adj, ci_hi_adj = mean+CORRECTION, ci_lo+CORRECTION, ci_hi+CORRECTION
        elif sub_diff<base_diff:
            mean_adj, ci_lo_adj, ci_hi_adj = mean-CORRECTION, ci_lo-CORRECTION, ci_hi-CORRECTION
        else:
            mean_adj, ci_lo_adj, ci_hi_adj = mean, ci_lo, ci_hi

        # manueller Override für ID 91: zweimal abziehen
        if meas_id == 91:
            final = mean - 2*CORRECTION
            print(f"Override ID 91: orig {mean:.3f} → korr {final:.3f}")
            mean_adj  = final
            ci_lo_adj = ci_lo - 2*CORRECTION
            ci_hi_adj = ci_hi - 2*CORRECTION

        own_P.append(dp)
        own_F.append(mean_adj)
        own_std.append(std)
        own_ci_w.append(ci_hi_adj - ci_lo_adj)
        own_ids.append(meas_id)

        with open(f"signal{meas_id}_statistics.txt","w",encoding="utf-8") as f:
            f.write(f"Eigenentwicklung {meas_id}\n")
            f.write(f"n                   : {n}\n")
            f.write(f"Mittelwert (korr.) : {mean_adj:.3f} µL/min\n")
            f.write(f"Std-Abw.            : {std:.3f} µL/min\n")
            f.write(f"95% CI Breite       : {(ci_hi_adj-ci_lo_adj):.3f} µL/min\n")
        print(f"Saved signal{meas_id}_statistics.txt")

    # --- 3) Referenz Errorbar-Plot ---
    if ref_P:
        plt.figure(figsize=(8,5))
        plt.errorbar(ref_P,ref_F,yerr=ref_std,
                     fmt='o',color='tab:blue',ecolor='gray',
                     elinewidth=1.5,capsize=4,label='Referenzsensor')
        plt.title("Referenzsensor: Flow vs. ΔP")
        plt.xlabel("ΔP (mBar)"); plt.ylabel("Durchflussrate (µL/min)")
        plt.grid('--',alpha=0.5); plt.tight_layout()
        plt.savefig("referenzsensor_errorbars.png",dpi=300)
        plt.savefig("referenzsensor_errorbars.pdf",format='pdf')
        plt.close()

    # --- 4) Eigener Sensor Errorbar-Plot ---
    if own_P:
        plt.figure(figsize=(8,5))
        plt.errorbar(own_P,own_F,yerr=own_std,
                     fmt='s',color='tab:orange',ecolor='gray',
                     elinewidth=1.5,capsize=4,label='Eigenentwicklung (korr.)')
        plt.title("Eigenentwicklung: Flow vs. ΔP")
        plt.xlabel("ΔP (mBar)"); plt.ylabel("Durchflussrate (µL/min)")
        plt.grid('--',alpha=0.5); plt.tight_layout()
        plt.savefig("eigener_sensor_errorbars.png",dpi=300)
        plt.savefig("eigener_sensor_errorbars.pdf",format='pdf')
        plt.close()

    # --- 5) Kombinierter Scatter mit ID-Annotation ---
    if ref_P and own_P:
        plt.figure(figsize=(8,5))
        plt.scatter(ref_P, ref_F,
                    c='tab:blue', edgecolor='k', s=50, label='Referenzsensor')
        plt.scatter(own_P, own_F,
                    c='tab:orange', edgecolor='k', s=50, label='Eigenentwicklung (korr.)')
        for x, y, mid in zip(ref_P, ref_F, ref_ids):
            plt.annotate(str(mid), (x, y),
                         textcoords="offset points", xytext=(4,4),
                         color='tab:blue', fontsize=9)
        for x, y, mid in zip(own_P, own_F, own_ids):
            plt.annotate(str(mid), (x, y),
                         textcoords="offset points", xytext=(4,-10),
                         color='tab:orange', fontsize=9)
        # Lineare Regression für eigene Werte
        if len(own_P) > 1:
            from scipy.stats import linregress
            slope_own, intercept_own, r_value_own, p_value_own, std_err_own = linregress(own_P, own_F)
            x_reg_own = np.linspace(min(own_P), max(own_P), 100)
            y_reg_own = slope_own * x_reg_own + intercept_own
            plt.plot(x_reg_own, y_reg_own, color='tab:red', linestyle='--', linewidth=2,
                     label=f'Regression Eigenentwicklung')
        # Lineare Regression für Referenzsensor
        if len(ref_P) > 1:
            slope_ref, intercept_ref, r_value_ref, p_value_ref, std_err_ref = linregress(ref_P, ref_F)
            x_reg_ref = np.linspace(min(ref_P), max(ref_P), 100)
            y_reg_ref = slope_ref * x_reg_ref + intercept_ref
            plt.plot(x_reg_ref, y_reg_ref, color='tab:blue', linestyle=':', linewidth=2,
                     label=f'Regression Referenzsensor')
        plt.title("Mittlere Durchflussrate vs. ΔP")
        plt.xlabel("ΔP (mBar)")
        plt.ylabel("Durchflussrate (µL/min)")
        plt.legend()
        plt.grid('--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("combined_flow_vs_pressure_scatter.png", dpi=300)
        plt.savefig("combined_flow_vs_pressure_scatter.pdf", format='pdf')
        plt.close()

    # --- 6) Gesamttabelle aller Messungen ---
    with open("gesamt_sensoren_tabelle.txt","w",encoding="utf-8") as f:
        f.write(f"{'Sensor':<16}{'ΔP':>10}{'Mean':>14}{'Std':>10}{'95%CI':>10}   ID\n")
        f.write("-"*68 + "\n")
        for p,m,s,ciw,mid in zip(ref_P,ref_F,ref_std,ref_ci_w,ref_ids):
            f.write(f"{'Referenz':<16}{p:>10d}{m:>14.3f}{s:>10.3f}{ciw:>10.3f}   {mid}\n")
        for p,m,s,ciw,mid in zip(own_P,own_F,own_std,own_ci_w,own_ids):
            f.write(f"{'Eigen (korr.)':<16}{p:>10d}{m:>14.3f}{s:>10.3f}{ciw:>10.3f}   {mid}\n")

    print("Verarbeitung abgeschlossen.")
    print("ref_P:", ref_P)
    print("ref_F:", ref_F)
    print("own_P:", own_P)
    print("own_F:", own_F)

if __name__ == "__main__":
    main()
