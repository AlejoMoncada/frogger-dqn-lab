# frogger-dqn-lab

> 🚧 **Trabajo en curso — Taller 2 (U3), Grupo 2.** El README final (espacios de observación y
> acción, recompensas, flujo de entrenamiento, arquitectura, resultados y reflexión) se completa
> antes de la entrega del lunes 12 de octubre.

DQN implementado desde cero con **PyTorch + Gymnasium** (sin Stable-Baselines3) sobre
`ALE/Frogger-v5`, usando la observación RAM de la consola (128 bytes) y una red MLP.

Este repositorio es independiente del taller de PPO (`frogger-rl-lab`).

## Estado actual

- ✅ Agente DQN: replay buffer, ε-greedy, red target, actualización de Bellman (pérdida Huber).
- ✅ Entrenamiento base de 300 000 pasos en CPU (~6 min): puntaje greedy 25.2 ± 4.4 vs. 9.4 ± 2.1
  de una política aleatoria (`results/baseline_300k/`).
- 🔄 Sistema de hitos/medallas por checkpoint con capturas y GIFs.
- ⏳ Entrenamiento largo, README final.

## Uso rápido

```bash
uv sync
uv run pytest -q
uv run python train.py --smoke --output-dir runs/smoke   # prueba rápida del pipeline
uv run python train.py --output-dir results/run_main     # entrenamiento completo
```
