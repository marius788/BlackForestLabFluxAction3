# NexArm-Aufnahme für FLUX-3-Action

Dieser Ordner nimmt NexArm-Daten so auf, dass sie zum SO-101-Vertrag von FLUX-3-Action passen:

| | Alter Weg (`..\finsh`, Conda `nexarm`) | Dieser Ordner |
|---|---|---|
| LeRobot | Hiwonders Fork | Upstream am Commit `2f05d8f` + Plugin `lerobot_robot_nexarm` |
| `action` / `observation.state` | Servo-Rohwerte (ca. 1000–3445) | Grad, Greifer 0–100 |
| Kalibrierung | Platzhalter `None.json` | gemessen, in `calibration\` |
| Kameras | `scene`, `wrist` | `scene`, `wrist` (unverändert) |

Die alte Umgebung bleibt unangetastet. Für F3A-Aufnahmen nur noch die Skripte hier benutzen.

## Reihenfolge

Alles per Doppelklick. Immer nur ein Fenster gleichzeitig an den Armen.

1. **`1-calibrate-follower.cmd`** – Arm abstützen (Drehmoment geht aus). Mittelpose einnehmen,
   ENTER. Dann jedes Gelenk nacheinander langsam durch den ganzen Bereich bewegen, jeweils ENTER.
2. **`2-calibrate-leader.cmd`** – derselbe Ablauf am Leader.
3. **`3-teleop-no-cameras.cmd`** – prüfen, ob jedes Gelenk in die richtige Richtung folgt.
4. **`4-teleop.cmd`** – Teleoperation mit Kameras und Live-Ansicht.
5. **`5-record.cmd`** – nimmt 20 Episoden nach `local/nexarm_f3a_stack_boxes_<Zeitstempel>` auf.
6. **`check-dataset.cmd`** – muss `OK` melden. `FAIL` heißt: nicht weiter aufnehmen.
7. **`6-record-continue.cmd`** – setzt eine unterbrochene Aufnahme bis zum selben Ziel von 20 Episoden fort.

`shell.cmd` öffnet eine Eingabeaufforderung mit aktiver Umgebung, z. B. für
`lerobot-find-port` oder `lerobot-find-cameras opencv`.

## Kalibrierung: darauf kommt es an

Die Kalibrierung legt fest, was „0 Grad“ ist und in welche Richtung positiv gezählt wird.
Davon hängt ab, ob die Daten zum SO-101-Checkpoint passen.

- **Nullpunkt:** Die Pose beim ersten ENTER wird 0 Grad für alle fünf Armgelenke. Nehmt dafür
  die SO-101-Mittelpose (jedes Gelenk in der Mitte seines Bereichs), bei beiden Armen gleich.
- **Richtungen:** Das Plugin misst sie nicht. In `calibration\follower\follower.json` steht pro
  Gelenk `drive_mode`: `0` = Vorzeichen +, `1` = Vorzeichen −. Beim Greifer dreht `1` die Skala um
  (`0` muss geschlossen sein, `100` offen). Nach einer Änderung beide Arme neu starten.
- **Greifer:** Der gemessene Bereich des Followers muss innerhalb der Rohwerte 1195–2833 liegen,
  sonst lehnt das Plugin die Datei ab.
- Der Leader rechnet mit der Follower-Datei. Richtungen deshalb nur dort ändern.
- Falls der Arm schon an einem anderen Rechner mit dem Plugin kalibriert wurde: `follower.json`
  und `leader.json` von dort nach `calibration\follower\` bzw. `calibration\leader\` kopieren.

`check-dataset.cmd` zeigt pro Gelenk den aufgenommenen Bereich neben dem Bereich, den der
F3A-SO-101-Checkpoint im Training gesehen hat. Liegt ein Gelenk völlig daneben oder ist es
gespiegelt, stimmt Nullpunkt oder `drive_mode` nicht.

## Einstellungen

- **COM-Ports, Datensatzname, Zielanzahl:** oben in `_env.cmd` (aktuell Follower `COM10`,
  Leader `COM15`).
- **Kameras, Aufgabe, Episodenlänge:** `config\record.yaml` (Kamera 0 = `scene`, 2 = `wrist`).
- **Datensätze:** liegen wie bisher auf Google Drive unter `HF_LEROBOT_HOME\local\`.
  Die Kalibrierung liegt lokal in `calibration\`.

## Neu installieren

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Das klont das Plugin nach `plugin\`, installiert Python 3.12 (x64) und die Umgebung nach
`uv.lock` und lässt die Plugin-Tests laufen. Hiwonders Fork darf nicht in diese Umgebung.

## Grenzen

- Grad und 0–100 machen die Daten formal F3A-kompatibel. Ob ein SO-101-Checkpoint auf dem
  NexArm geometrisch funktioniert, ist damit nicht belegt; ein Finetuning auf den eigenen
  Daten ist der vorgesehene Weg.
- Die alten Rohwert-Aufnahmen (`nexarm_stack_boxes_*`) lassen sich nicht sauber umrechnen,
  weil dafür nie ein Nullpunkt gemessen wurde.
