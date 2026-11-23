# NexArm-Entwicklungsumgebung auf Windows ARM

Dieses Verzeichnis enthaelt ein reproduzierbares Setup fuer die Hiwonder-NexArm-LeRobot-Anleitung.

## Architekturentscheidung

Der Laptop laeuft mit Windows 11 ARM64. Fuer NexArm wird absichtlich **Miniconda x86-64** in der Windows-x64-Emulation verwendet. Das entspricht dem von Hiwonder getesteten Windows-Pfad und bietet die vollstaendigere Wheel-Unterstuetzung fuer PyTorch, OpenCV, TorchCodec und Rerun. Eine vorhandene globale Python-Installation wird nicht veraendert.

Installiert werden:

- Miniconda x86-64 aus der Hiwonder-Anleitung
- isolierte Conda-Umgebung `nexarm` mit Python 3.12
- offizieller Hiwonder-Fork `lerobot-nexarm`
- NexArm-/Hardware-, Dataset-, Visualisierungs- und Trainingsabhaengigkeiten
- FFmpeg 7.1.1
- PyTorch (CPU unter Windows ARM/x64-Emulation)

## Installation

In einer normalen PowerShell in diesem Ordner ausfuehren:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-nexarm.ps1
```

Danach kann `open-nexarm-shell.cmd` per Doppelklick geoeffnet werden. Die Shell startet direkt im Repository mit aktivierter `nexarm`-Umgebung.

## Pruefung

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\verify-nexarm.ps1
```

Der Softwaretest muss `NEXARM_OK` ausgeben.

## Hardwaretest morgen

1. Zuerst den Follower-Arm anschliessen.
2. `open-nexarm-shell.cmd` starten.
3. Port ermitteln:

   ```powershell
   python -m lerobot.scripts.lerobot_find_port
   ```

4. Danach den Leader-Arm genauso ermitteln.
5. Beide Kameras anschliessen und IDs pruefen:

   ```powershell
   python -m lerobot.scripts.lerobot_find_cameras opencv
   ```

6. Falls kein CH340/CH341-COM-Port erscheint, den signierten Windows-Treiber von WCH installieren: <https://www.wch-ic.com/downloads/CH341SER_ZIP.html>. Der aktuelle Treiber unterstuetzt Windows 11 ARM64.

Die beiden Kameras laut Hiwonder nicht gleichzeitig an denselben USB-Hub/Docking-Station-Zweig anschliessen.

## Einschraenkung des ARM-Laptops

Lokales CUDA-Training ist nicht moeglich, weil CUDA eine NVIDIA-GPU voraussetzt. Teleoperation, Datenerfassung, Visualisierung und CPU-Inferenz funktionieren ueber die x64-Emulation. Fuer umfangreiches Training sollte spaeter ein NVIDIA-Rechner oder Cloud-GPU-System verwendet werden.

