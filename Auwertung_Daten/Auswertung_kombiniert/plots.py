import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# --- 1) Daten einlesen ---
# Wir gehen davon aus, dass „gesamt_sensoren_tabelle.txt“ im selben Ordner liegt
# und die Trennlinie aus Bindestrichen in Zeile 2 übersprungen werden muss.
df = pd.read_csv(
    'gesamt_sensoren_tabelle.txt',
    sep=r'\s{2,}',        # Spalten werden durch 2+ Leerzeichen getrennt
    engine='python',
    skiprows=[1]          # Zeile mit '----…----'
)

# Spalten umbenennen für einfachere Verarbeitung
df = df.rename(columns={
    'ΔP':    'DeltaP',
    'Mean':  'Mean',
    'Std':   'StdDev',
    'ID':    'ID',
    'Sensor':'Sensor'
})

# Numerische Konvertierung
for col in ['DeltaP', 'Mean', 'StdDev', 'ID']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- 2) Fehlerbalkendiagramm (Scatter + y-Errorbars) pro Sensor ---
out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

for sensor_name, group in df.groupby('Sensor'):
    plt.figure(figsize=(8,5))
    # reine Scatter-Punkte (fmt='o' ohne Linienstil) mit Fehlerbalken
    plt.errorbar(
        group['DeltaP'],
        group['Mean'],
        yerr=group['StdDev'],
        fmt='o',
        capsize=4,
        label=sensor_name
    )
    plt.title(f'Durchflussrate ± StdDev – {sensor_name}')
    plt.xlabel('Druckdifferenz ΔP (mBar)')
    plt.ylabel('Durchflussrate (µL/min)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    # Safe filename: Kleinbuchstaben, Unterstriche, keine Klammern
    fn = re.sub(r'[^a-z0-9_]+', '_',
                sensor_name.lower().replace(' ', '_'))
    plt.savefig(f'{out_dir}/scatter_errorbars_{fn}.png', dpi=300)
    plt.close()

print(f"Plots wurden im Ordner '{out_dir}' abgelegt.")

# --- 3) Standardabweichungen in Textdatei speichern ---
std_file = 'sensor_standardabweichungen.txt'
with open(std_file, 'w', encoding='utf-8') as f:
    f.write("Sensor\tID\tΔP (mBar)\tStdDev (µL/min)\n")
    f.write("-"*45 + "\n")
    for _, row in df.iterrows():
        f.write(f"{row['Sensor']}\t"
                f"{int(row['ID'])}\t"
                f"{int(row['DeltaP'])}\t"
                f"{row['StdDev']:.3f}\n")

print(f"Standardabweichungen wurden in '{std_file}' gespeichert.")
