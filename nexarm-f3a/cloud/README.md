# FLUX 3 Action auf AI Galaxy (`gpu.ai-galaxy.com`) + MuJoCo-Simulation

Dieser Ordner integriert gezielt die fehlenden FLUX-3-Action-Bausteine (`server.py`, `client.py`, `sim.py`, `replay_eval.py`, `setup_remote.sh`) aus `mariorch22/lerobot-nexarm` (Apache-2.0) in dein Setup und passt das Deployment auf **AI Galaxy (智星云)** an.

---

## 1. Hugging-Face-Freigaben & Token (einmalig)

1. Auf **https://huggingface.co** einloggen (nicht über `hf-mirror.com`).
2. Bei **beiden** gated Repos die Lizenz akzeptieren (*Agree and access repository*):
   - <https://huggingface.co/black-forest-labs/flux-3-action-so101>
   - <https://huggingface.co/black-forest-labs/flux-3-action-base>
3. Unter <https://huggingface.co/settings/tokens> einen **Read-Token** (`hf_...`) erstellen.

---

## 2. AI Galaxy Instanz & Speicher richtig wählen

- **Empfohlene GPU:**
  - **NVIDIA A800-SXM4-40GB (40G):** Allrounder & günstig für Inferenz + LoRA-Training (`--batch-size 2`). Direkt nach dem Booten `nvidia-smi` prüfen: Host-CUDA muss **>= 12.8** (Treiber >= 570) sein! Falls `< 12.8`, sofort mit **结束** beenden und auf RTX 5090 wechseln.
  - **GeForce RTX 5090 (32G):** Schnellste Live-Inferenz, Treiber >= 12.8 garantiert. Für LoRA-Training `--batch-size 1` nutzen (bei Inferenz optional `--text-encoder-on-cpu`).
- **Disk:** mindestens **80–120 GB** wählen. Nach dem ersten SSH-Login mit `df -h && lsblk` prüfen, wo der große Speicher hängt (oft `/root/data`, `/data` oder `/workspace`), und diesen Pfad (z. B. `/root/data/nexarm`) als `REMOTE_DIR` in `cloud/.env` eintragen.
- **Kostenfalle:** Bei Minutenabrechnung stoppt auf AI Galaxy **nur `结束` (End/Release)** die Abrechnung — ein normales **`关机` (Shut Down)** läuft kostenpflichtig weiter!

---

## 3. SSH-Key einmalig einrichten & `cloud/.env` ausfüllen

1. `nexarm-f3a\cloud\.env.example` nach `nexarm-f3a\cloud\.env` kopieren (wird per `.gitignore` ignoriert) und `SSH_HOST`, `SSH_PORT`, `REMOTE_DIR`, `HF_TOKEN` eintragen.
2. Einmalig in PowerShell einen SSH-Key anlegen (falls noch nicht vorhanden) und mit dem Start-Passwort aus der AI-Galaxy-Konsole hochladen:

```powershell
if (!(Test-Path "$env:USERPROFILE\.ssh\id_ed25519")) { ssh-keygen -t ed25519 -N '""' -f "$env:USERPROFILE\.ssh\id_ed25519" }
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | ssh -p <SSH_PORT> root@<SSH_HOST> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

*(Falls AI Galaxy intern einen HF-Mirror anbietet, kannst du ihn alternativ testen; Standard in `.env.example` ist `HF_ENDPOINT=https://hf-mirror.com` mit `HF_HUB_DISABLE_XET=1` für `huggingface-hub==2.4.0`.)*

---

## 4. Ablauf: Setup, Server, Dry-Run, Execute & LoRA-Training

Alle Befehle aus dem Ordner `nexarm-f3a\` (über `shell.cmd`):

```powershell
# 1. Status / SSH / GPU / freien Speicher prüfen
python cloud\galaxy.py status

# 2. Umgebung installieren & ~25 GB Gewichte herunterladen (auf CPU-Host mit --cpu)
python cloud\galaxy.py setup

# 3. Server in tmux starten & SSH-Tunnel nach http://127.0.0.1:8000 öffnen
python cloud\galaxy.py up
# (Mit trainiertem Adapter: python cloud\galaxy.py up --adapter outputs\lora\nexarm_lora\pretrained_model)
# (Auf 32 GB GPU optional: --text-encoder-on-cpu)
```

In einem zweiten Fenster (`shell.cmd` oder Doppelklick auf die `.cmd`-Launcher):

```powershell
# 4. Tunnel & Server ohne Arm testen
python cloud\client.py --fake-robot --task "put the small box on top of the big box"

# 5. Dry-Run am echten Arm (sendet KEINE Bewegungen, prüft Kamera/Kalibrierung/Pläne)
7-inference-dryrun.cmd

# 6. Echte Bewegung (erst nach erfolgreichem Dry-Run & gesetzten Limits!)
8-inference-execute.cmd
```

### LoRA-Training auf deinen Aufnahmen & Offline-Auswertung

```powershell
# A800 (40G, Default batch_size=2):
python cloud\galaxy.py train --dataset "H:\Meine Ablage\02_ARBEIT_&_PROJEKTE\FluxAction\lerobot\local\<DEIN_DATASET>"

# RTX 5090 (32G):
python cloud\galaxy.py train --dataset "..." --batch-size 1

python cloud\galaxy.py logs
python cloud\galaxy.py curves
python cloud\galaxy.py fetch

# Offline-Vergleich gegen zurückgehaltene Episoden (wenn Server über `up` läuft):
python cloud\replay_eval.py --dataset "H:\...\<DEIN_DATASET>" --episodes 14 15 16 17
```

---

## 5. MuJoCo-Simulation (`sim.py`)

Zum gefahrlosen Testen der Regelschleife mit Physik & gerenderten Bildern (ohne Hardware):

```powershell
git clone --depth 1 https://github.com/khanhthanhdev/lerobot-nexarm.git third_party\khanhthanhdev-lerobot-nexarm
git -C third_party\khanhthanhdev-lerobot-nexarm lfs pull --include "sim/assets/**"
```

Danach per Doppelklick `sim-test.cmd` (speichert Video nach `outputs\sim.mp4`).
Für einen reinen Bewegungstest **ohne GPU** lokal:

```powershell
uv run cloud\server.py --mock-sweep
# Im zweiten Fenster (Greifer-Chunk-Span für Sweep auf 0 setzen):
sim-test.cmd --chunk-gripper-span 0
```
