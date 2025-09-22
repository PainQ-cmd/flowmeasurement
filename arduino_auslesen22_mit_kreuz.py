# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:43:09 2025

@author: chr5573s

Anpassung: t0_1, t0_2, dt_1, dt_2 sowie Zero-Crossing-Zeiten und Kreuzkorrelation von wf1 und wf2 in Millisekunden mit 16 Dezimalstellen ausgegeben und gespeichert
"""

import os
import csv
import serial
import numpy as np
import matplotlib.pyplot as plt
import time
from DSO import DSO
from zero_crossing import ZeroCrossingTracker

# Anzahl der Nachkommastellen für ms-Werte
NUM_DECIMALS = 16

plt.rcParams['toolbar'] = 'toolbar2'
plt.ion()

# Erstelle eine DSO-Instanz und verbinde dich mit dem Gerät
dso = DSO()
device_address = "TCPIP0::isat-croy11::inst0::INSTR"

try:
    dso.connect(device_address)
    print(f"Erfolgreich mit {device_address} verbunden!")
except Exception as e:
    print(f"Fehler beim Verbinden mit {device_address}: {e}")
    exit()

# COM-Port öffnen
arduinoData = serial.Serial('COM7', baudrate=115200, timeout=1)
print("COM-Port geöffnet für Arduino-Daten")

elektronik = serial.Serial('COM8', 256000, timeout=1)
command = "LTX\n"
elektronik.write(command.encode())
response = elektronik.readline().decode().strip()
print(response)

# Datenarrays für Live-Plot
temp = []
flow = []
elapsed = []

# Plot-Initialisierung
fig, (ax_temp, ax_flow, ax_dso) = plt.subplots(3, 1, figsize=(10, 12))
line_temp, = ax_temp.plot([], [], 'r-', label="Temperature (°C)")
line_flow, = ax_flow.plot([], [], 'b-', label="Flow Rate (µl/min)")
line_dso, = ax_dso.plot([], [], 'g-', label="Waveform", alpha=0.7)

# Plot-Konfiguration
ax_temp.set(title="Live Temperatur", xlabel="Zeit (s)", ylabel="°C")
ax_flow.set(title="Live Flow Rate", xlabel="Zeit (s)", ylabel="µl/min")
ax_dso.set(title="Oszilloskop-Signal", xlabel="Zeit (s)", ylabel="Spannung (V)")
for ax in [ax_temp, ax_flow, ax_dso]:
    ax.legend(loc="upper right")
    ax.grid(True)
plt.tight_layout()

# Variablen für Zero-Crossing-Tracking
prev_zc_pos = np.array([])
prev_zc_type = np.array([])

# Nächst möglichen Dateinamen ermitteln
def get_next_filename(base_name="measurement", extension=".txt"):
    count = 1
    while os.path.exists(f"{base_name}{count}{extension}"):
        count += 1
    return f"{base_name}{count}{extension}"

filename = get_next_filename()
print(f"Speichere Daten in: {filename}")

start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
start_time = None
count = 0

with open(filename, "w", newline='', encoding='utf-8') as f, \
     open("signal1.txt", "w") as file_1, \
     open("signal2.txt", "w") as file_2:
    writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    
    # Header schreiben (Messungsstart, Spaltenbeschriftungen)
    writer.writerow([f"Messung gestartet am: {start_timestamp}"])
    writer.writerow([
        "Zeitstempel (s)",
        "Flow Rate (µl/min)",
        "Temperatur (°C)",
        "Anzahl Nulldurchgänge",
        "Zero-Crossing-Zeiten (ms)",
        "t0_1 (ms)",
        "t0_2 (ms)",
        "dt_1 (ms)",
        "dt_2 (ms)",
        "Max Kreuzkorrelation",
        "Lag bei Max (ms)"
    ])

    while count < 50:
        # Arduino-Daten lesen
        while arduinoData.inWaiting() == 0:
            pass
        arduinoString = arduinoData.readline().decode('utf-8').strip()
        try:
            dataArray = arduinoString.split(',')
            flowRate = float(dataArray[0].strip())
            tempC = float(dataArray[1].strip())
            print(f"Daten empfangen: Flow={flowRate} µl/min, Temp={tempC} °C")
        except (IndexError, ValueError) as e:
            print(f"Fehler beim Verarbeiten der Daten: {e} - Empfangene Zeichenkette: {arduinoString}")
            continue
        
        # Startzeit setzen
        if start_time is None:
            start_time = time.time()
            print("Messung gestartet")

        # Zeit seit Messungsstart
        elapsed_time = time.time() - start_time
        elapsed.append(elapsed_time)
        flow.append(flowRate)
        temp.append(tempC)

        # Oszilloskop-Daten abrufen und verarbeiten
        zc_count = 0
        zc_times_str = ""
        try:
            # Kanal 1 aufnehmen
            elektronik.write(b"Mim;255,1,0,1500\n")
            elektronik.readline()
            wf1 = dso.waveform('C1')
            t0_1, dt_1 = dso.waveform_t0_dt('C1')

            # Kanal 2 aufnehmen
            elektronik.write(b"Mim;255,0,1,1500\n")
            elektronik.readline()
            wf2 = dso.waveform('C1')
            t0_2, dt_2 = dso.waveform_t0_dt('C1')

            # Waveforms speichern
            file_1.write(",".join(map(str, wf1)) + "\n")
            file_2.write(",".join(map(str, wf2)) + "\n")

            # Zero-Crossing Tracking basierend auf Kanal 1
            time_axis = np.arange(len(wf1)) * dt_1 + t0_1
            if count == 0:
                current_zc_pos, current_zc_type = ZeroCrossingTracker.calc_wf_zero_crossings(wf1)
            else:
                current_zc_pos, current_zc_type = ZeroCrossingTracker.track_zero_crossings(
                    wf1, prev_zc_pos, prev_zc_type)
            prev_zc_pos, prev_zc_type = current_zc_pos, current_zc_type

            # Absolute ZC-Zeiten in ms
            zc_times = t0_1 + current_zc_pos * dt_1
            zc_times_ms = zc_times * 1000
            zc_count = len(zc_times_ms)
            zc_times_str = ";".join([f"{tm:.{NUM_DECIMALS}f}" for tm in zc_times_ms])

            print(f"Nulldurchgänge bei (ms): {[round(tm, NUM_DECIMALS) for tm in zc_times_ms]}")
            print(f"t0_1 = {t0_1*1000:.{NUM_DECIMALS}f} ms, dt_1 = {dt_1*1000:.{NUM_DECIMALS}f} ms")
            print(f"t0_2 = {t0_2*1000:.{NUM_DECIMALS}f} ms, dt_2 = {dt_2*1000:.{NUM_DECIMALS}f} ms")

            # Kreuzkorrelation von wf1 und wf2
            wf1_zm = wf1 - np.mean(wf1)
            wf2_zm = wf2 - np.mean(wf2)
            cross_corr = np.correlate(wf1_zm, wf2_zm, mode='full')
            max_idx = np.argmax(cross_corr)
            lags = np.arange(-len(wf1_zm)+1, len(wf2_zm))
            lag_samples = lags[max_idx]
            lag_time = lag_samples * dt_1  # Sekunde
            max_corr = cross_corr[max_idx]
            print(f"Maximale Kreuzkorrelation: {max_corr:.{NUM_DECIMALS}f}, bei Versatz {lag_time*1000:.{NUM_DECIMALS}f} ms")

            # Plot-Aktualisierung für Kanal 1
            line_dso.set_data(time_axis, wf1)
            ax_dso.set_xlim(time_axis[0], time_axis[-1])
            ax_dso.set_ylim(np.min(wf1)*1.1, np.max(wf1)*1.1)
            ax_dso.scatter(zc_times, np.zeros_like(zc_times), marker='x', label='Zero-Crossings')

        except Exception as e:
            print(f"DSO-Fehler: {e}")

        # Live-Plots aktualisieren
        line_temp.set_data(elapsed, temp)
        line_flow.set_data(elapsed, flow)
        ax_temp.set_xlim(0, max(elapsed))
        ax_flow.set_xlim(0, max(elapsed))
        ax_temp.set_ylim(min(temp)-1, max(temp)+1)
        ax_flow.set_ylim(min(flow)-0.1, max(flow)+0.1)
        plt.pause(0.01)

        # Daten in die Datei schreiben
        writer.writerow([
            f"{elapsed_time:.3f}",  # Zeitstempel (s)
            flowRate,               # Flow Rate (µl/min)
            tempC,                  # Temperatur (°C)
            zc_count,               # Anzahl der Nulldurchgänge
            zc_times_str,           # Zero-Crossing-Zeiten (ms)
            f"{t0_1*1000:.{NUM_DECIMALS}f}",  # t0_1 in ms
            f"{t0_2*1000:.{NUM_DECIMALS}f}",  # t0_2 in ms
            f"{dt_1*1000:.{NUM_DECIMALS}f}",  # dt_1 in ms
            f"{dt_2*1000:.{NUM_DECIMALS}f}",  # dt_2 in ms
            f"{max_corr:.{NUM_DECIMALS}f}",     # Max Kreuzkorrelation
            f"{lag_time*1000:.{NUM_DECIMALS}f}" # Lag bei Max (ms)
        ])

        count += 1

print(f"{count} Messungen in '{filename}' gespeichert.")

# Verbindungen schließen
arduinoData.close()
elektronik.close()
print("COM-Ports geschlossen.")
dso.disconnect()
print("Oszilloskop-Verbindung getrennt.")
file_1.close()
file_2.close()
plt.ioff()
plt.show()
