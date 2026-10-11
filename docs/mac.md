# Entrenamiento en Mac (Apple Silicon, MPS)

Guía para ejecutar el DQN con píxeles en un MacBook Pro M4 (16 GB) usando la GPU de Apple (MPS).

## Notas previas

- Mantén la tapa abierta y el equipo conectado a la corriente durante todo el entrenamiento.
- 16 GB de memoria unificada bastan: el replay de frames con 200k transiciones ocupa unos 1.4 GB.
- La velocidad de MPS es desconocida hasta medir el piloto de 100k pasos. Compárala con la de la
  tabla de `timing.json` antes de planear el run largo.
- MPS aún no está verificado en este proyecto: la primera ejecución real es el piloto.

## 1. Instalar uv

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Cierra y abre la terminal, o ejecuta `source $HOME/.local/bin/env`.

## 2. Clonar la rama

```sh
git clone --branch exp/home-challenge https://github.com/AlejoMoncada/frogger-dqn-lab.git
cd frogger-dqn-lab
```

## 3. Instalar dependencias

```sh
uv sync
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

La segunda línea debe imprimir `True`. En macOS, `uv` instala PyTorch desde PyPI (no desde el
índice CUDA de Windows/Linux).

## 4. Pruebas y smoke test

```sh
uv run pytest -q
uv run python train_pixels.py --smoke --device mps --output-dir runs/mac_smoke
```

## 5. Piloto de 100k pasos

```sh
uv run python train_pixels.py --total-steps 100000 --device mps --output-dir runs/mac_pilot
```

Lee `runs/mac_pilot/timing.json`:

- `device`: debe ser `"mps"`.
- `host.machine` y `host.macos_version`: confirman el equipo y la versión de macOS.
- `steps_per_second`: velocidad total del run; úsala para estimar el tiempo del run largo.
- `time_breakdown_seconds`: reparte el tiempo entre `env_step`, `action_selection`,
  `learner_update` y `eval`.

## 6. Run 3 replicado con semilla 43

Mantén el equipo despierto con `caffeinate`:

```sh
caffeinate -dimsu uv run python train_pixels.py --total-steps 5000000 --device mps \
  --buffer-size 200000 --reward-mode scale --reward-scale 0.1 --seed 43 \
  --output-dir results/run_px_5m_r3_seed43_mac
```

Opcional (experimental, H12): añade `--life-loss-terminal` para tratar la pérdida de una vida como fin del arranque TD; está desactivado por defecto.

Si el proceso se interrumpe, reanúdalo en el mismo directorio (usa `checkpoints/latest.pt`). Repite
siempre `--total-steps 5000000`: si se omite, el valor por defecto es 2 000 000 y el run terminaría antes.

```sh
caffeinate -dimsu uv run python train_pixels.py --resume --total-steps 5000000 --device mps \
  --buffer-size 200000 --reward-mode scale --reward-scale 0.1 --seed 43 \
  --output-dir results/run_px_5m_r3_seed43_mac
```

## 7. Leaderboard y curva al final

```sh
uv run python milestones.py --run-dir results/run_px_5m_r3_seed43_mac --episodes 20
uv run python plot.py --input-dir results/run_px_5m_r3_seed43_mac
```
