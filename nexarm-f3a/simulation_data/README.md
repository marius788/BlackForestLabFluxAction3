# Simulation Data – Kamera-Kalibrierung & Real-vs-Sim-Vergleiche

Übersicht aller Referenzbilder, Messwerte, Vorher/Nachher-Standbilder und Vergleichsvideos (alle Videos in **H.264 / `yuv420p`**, direkt abspielbar in Windows Media Player, Browser, VLC und VS Code / Antigravity).

## Ordnerstruktur

```text
simulation_data/
├── 01_real_reference/              # Originalaufnahmen & Pixel-Statistiken aus V1SchachBauer_clean
│   ├── real_scene_frame0000.png
│   ├── real_wrist_frame0000.png
│   ├── real_scene_zoom4x.png
│   ├── real_wrist_zoom4x.png
│   ├── real_scene_calibration_grid.png
│   ├── real_wrist_calibration_grid.png
│   └── real_camera_stats.json
├── 02_before_calibration/          # Unkalibrierter MuJoCo-Stand (vor Kamera-/Brett-Anpassung)
│   ├── episode_000_real_vs_sim_raw.mp4
│   └── frames/
│       ├── frame_0000_start.png
│       ├── frame_0140_approach.png
│       ├── frame_0280_grasp.png
│       ├── frame_0420_placed.png
│       └── frame_0561_end.png
├── 03_calibrated_comparison/       # Kalibrierte Simulation (Fischauge, Brett, WB, LUT, Rauschen, AV1)
│   ├── episode_000_real_vs_sim_calibrated.mp4
│   ├── fit_report.json
│   └── frames/
│       ├── frame_0000_start.png
│       ├── frame_0140_approach.png
│       ├── frame_0280_grasp.png
│       ├── frame_0420_placed.png
│       └── frame_0561_end.png
└── 04_rollouts/                    # Einzelne Simulations-Rollouts (sim-test.cmd)
    └── sim_rollout.mp4
```

## Neue Vergleiche erzeugen

```powershell
uv run --project plugin --with mujoco python cloud\compare_sim_real.py `
  --dataset ..\local\V1SchachBauer_clean --episode 0 --scene ..\Schach\chess_scene.xml
```
