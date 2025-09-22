# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:43:09 2025

@author: chr5573s

Anpassung: t0, dt und Zero-Crossing-Zeiten werden in Millisekunden mit 4 Dezimalstellen ausgegeben und gespeichert
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
print("COM-Port geöffnet")

elektronik = serial.Serial('COM8', 256000, timeout=1)
command = "LTX\n"
elektronik.write(command.encode())
response = elektronik.readline().decode().strip()
print(response)

# Datenarrays
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
    
    # Header schreiben (t0, dt und ZC-Zeiten in ms)
    writer.writerow([f"Messung gestartet am: {start_timestamp}"])
    writer.writerow([
        "Zeitstempel (s)",
        "Flow Rate (µl/min)",
        "Temperatur (°C)",
        "Anzahl Nulldurchgänge",
        "Zero-Crossing-Zeiten (ms)",
        "t0 (ms)",
        "dt (ms)",
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
        
        if start_time is None:
            start_time = time.time()
            print("Messung gestartet")

        elapsed_time = time.time() - start_time
        elapsed.append(elapsed_time)
        flow.append(flowRate)
        temp.append(tempC)

        # Oszilloskop-Daten abrufen und verarbeiten
        zc_count = 0
        zc_times_str = ""
        try:
            # Mikrocode-Befehle für DSO
            elektronik.write(b"Mim;255,1,0,1500\n")
            elektronik.readline()
            wf = dso.waveform('C1')
            t0, dt = dso.waveform_t0_dt('C1')

            elektronik.write(b"Mim;255,0,1,1500\n")
            elektronik.readline()
            wf_2 = dso.waveform('C1')
            t0_2, dt_2 = dso.waveform_t0_dt('C1')

            # Waveforms speichern
            file_1.write(",".join(map(str, wf)) + "\n")
            file_2.write(",".join(map(str, wf_2)) + "\n")

            # Zero-Crossing Tracking
            time_axis = np.arange(len(wf)) * dt + t0
            if count == 0:
                current_zc_pos, current_zc_type = ZeroCrossingTracker.calc_wf_zero_crossings(wf)
            else:
                current_zc_pos, current_zc_type = ZeroCrossingTracker.track_zero_crossings(wf, prev_zc_pos, prev_zc_type)
            prev_zc_pos, prev_zc_type = current_zc_pos, current_zc_type

            # Absolute ZC-Zeiten in ms
            zc_times = t0 + current_zc_pos * dt
            zc_times_ms = zc_times * 1000
            zc_count = len(zc_times_ms)
            zc_times_str = ";".join([f"{tm:.{NUM_DECIMALS}f}" for tm in zc_times_ms])

            print(f"Nulldurchgänge erkannt bei (ms): {[round(tm, NUM_DECIMALS) for tm in zc_times_ms]}")
            print(f"t0 = {t0*1000:.{NUM_DECIMALS}f} ms, dt = {dt*1000:.{NUM_DECIMALS}f} ms")

            # Plot-Aktualisierung
            line_dso.set_data(time_axis, wf)
            ax_dso.set_xlim(time_axis[0], time_axis[-1])
            ax_dso.set_ylim(np.min(wf)*1.1, np.max(wf)*1.1)
            ax_dso.scatter(zc_times, np.zeros_like(zc_times), color='red', marker='x', label='Zero-Crossings')

        except Exception as e:
            print(f"DSO-Fehler: {e}")

        # Plots aktualisieren
        line_temp.set_data(elapsed, temp)
        line_flow.set_data(elapsed, flow)
        ax_temp.set_xlim(0, max(elapsed))
        ax_flow.set_xlim(0, max(elapsed))
        ax_temp.set_ylim(min(temp)-1, max(temp)+1)
        ax_flow.set_ylim(min(flow)-0.1, max(flow)+0.1)

        # Daten in die CSV schreiben
        writer.writerow([
            f"{elapsed_time:.3f}",  # Zeitstempel (s)
            flowRate,               # Flow Rate (µl/min)
            tempC,                  # Temperatur (°C)
            zc_count,               # Anzahl der Nulldurchgänge
            zc_times_str,           # ZC-Zeiten (ms)
            f"{t0*1000:.{NUM_DECIMALS}f}",  # t0 in ms
            f"{dt*1000:.{NUM_DECIMALS}f}"   # dt in ms
        ])
        
        plt.pause(0.01)
        count += 1

print(f"{count} Messungen in '{filename}' gespeichert.")

# Verbindungen schließen
arduinoData.close()
elektronik.close()
print("COM-Port wurde geschlossen.")
dso.disconnect()
print("Oszilloskop-Verbindung getrennt")

file_1.close()
file_2.close()
plt.ioff()
plt.show()
