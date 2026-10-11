"""Build the local story page: copy its GIFs and screenshots and write the embedded data file.

Reads every run folder under ``results/`` that has a ``leaderboard.json``, copies the GIFs and
screenshots that the leaderboard references into ``docs/historia/assets/<run>/``, and writes
``docs/historia/data.js`` as ``window.STORY_DATA = {...};``. The data is embedded in a script file
because ``fetch`` does not work from ``file://``, so the page opens by double-click.

The output is deterministic: the same inputs always produce the same ``data.js``, and the
``assets/`` folder is rebuilt from scratch on every run (stale files are removed).

Usage:
    uv run python tools/build_story.py
    uv run python tools/build_story.py --results results --out docs/historia
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_OUT = REPO_ROOT / "docs" / "historia"

MEDAL_ORDER: tuple[str, ...] = (
    "first_hop",
    "road_crossed",
    "first_log",
    "mid_river",
    "frog_home",
    "three_homes",
    "level_cleared",
)

# Medal ladder shown on the page. Row thresholds come from dqn_frogger/milestones.py
# (ROW_MEDALS); home thresholds from HOME_MEDALS and HOMES_FOR_LEVEL_CLEARED.
MEDAL_LADDER: list[dict[str, str]] = [
    {"key": "first_hop", "emoji": "🐸", "name": "Primer salto", "threshold": "fila 1"},
    {"key": "road_crossed", "emoji": "🥉", "name": "Cruce de la carretera", "threshold": "fila 6"},
    {"key": "first_log", "emoji": "🪵", "name": "Primer tronco", "threshold": "fila 7"},
    {"key": "mid_river", "emoji": "🥈", "name": "Mitad del río", "threshold": "fila 9"},
    {"key": "frog_home", "emoji": "🥇", "name": "Casa", "threshold": "1 casa en el episodio"},
    {"key": "three_homes", "emoji": "🏆", "name": "Tres casas", "threshold": "3 casas en el episodio"},
    {"key": "level_cleared", "emoji": "👑", "name": "Nivel completo", "threshold": "5 casas en el episodio"},
]

# Numbers that come from the task document and the development notes, not from result files.
# Each one cites its source so the page can state where it came from.
PILOT_H2: dict[str, Any] = {
    "source": "odd/tasks/home-challenge.md, H2 (100k pasos, 8 entornos)",
    "steps": 100000,
    "mixed_steps_per_second": 607.0,
    "all_cpu_steps_per_second": 122.8,
    "gpu_util_mean_percent": 45.6,
}
# Home detection status shown on the page. The leaderboards no longer mark home medals as proxies:
# the H6 step-by-step trace (1M checkpoint, seed 900008) matches the tracker rule in
# dqn_frogger/milestones.py (a positive reward on the 12th counted row, no life lost that step).
HOME_DETECTION: dict[str, str] = {
    "status": "verified",
    "evidence": (
        "Verificada con una traza paso a paso (checkpoint de 1 M, semilla 900008): al llegar a casa "
        "desde la fila 11, la recompensa es +55, las vidas siguen en 3 y la fila vuelve a 0."
    ),
}
MEMORY_NOTES: dict[str, Any] = {
    "source": "docs/notas-de-desarrollo.md, sección 7",
    "buffer_size": 200000,
    "stacked_gb": 11.3,
    "frame_gb": 1.42,
}

# Run folders in the order of the ablation grid (the RAM baseline first). Unknown folders are
# appended alphabetically with their folder name as label.
RUN_SPECS: list[dict[str, Any]] = [
    {
        "id": "run_2m",
        "label": "RAM + MLP (entrega)",
        "change": "Observación RAM de 128 bytes con una MLP. Es la entrega del taller, sin cambios.",
        "fallback_settings": {
            "observation": "RAM (128 bytes)",
            "network": "MLP 128-256-256-5",
            "buffer_size": 100000,
            "reward_mode": "sign",
            "seed": 42,
            "total_steps": 2000000,
            "source": "README.md (la corrida no tiene config.json)",
        },
    },
    {
        "id": "run_px_5m",
        "label": "Corrida 1",
        "change": "Píxeles y CNN, buffer de 200 000, recompensa con signo.",
        "partial": "Detenida por falta de memoria en 4,6 M de pasos (episodes.csv). Los checkpoints llegan hasta 3 M.",
    },
    {
        "id": "run_px_5m_safe",
        "label": "Corrida 2",
        "change": "Igual que la corrida 1, con buffer de 100 000.",
    },
    {
        "id": "run_px_5m_r3",
        "label": "Corrida 3",
        "change": "Buffer de 200 000 con cuadros compartidos, recompensa escalada por 0,1, semilla 42.",
        "partial": "Detenida por falta de memoria en 2,74 M de pasos; reanudada desde el checkpoint de 2,5 M.",
    },
    {
        "id": "run_px_5m_r4",
        "label": "Corrida 4",
        "change": "Buffer de 200 000 con cuadros compartidos y recompensa con signo.",
    },
    {
        "id": "run_px_5m_r5",
        "label": "Corrida 5",
        "change": "Igual que la corrida 4, con la pérdida de una vida como fin para el aprendizaje.",
    },
    {
        "id": "run_px_5m_r3_seed43_mac",
        "label": "Corrida 3b (Mac)",
        "change": "Réplica de la corrida 3 con semilla 43, en un MacBook Pro M4 (MPS).",
    },
]


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON object at ``path``, or None when the file is missing."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _title_for(row: dict[str, Any]) -> str:
    """Human title for a leaderboard row: 'Aleatoria' or the checkpoint step with spaces."""
    step = row.get("step")
    if step is None:
        return "Aleatoria"
    return f"{int(step):,}".replace(",", " ")


def _settings(
    spec: dict[str, Any], config: dict[str, Any] | None, timing: dict[str, Any] | None,
    leaderboard: dict[str, Any],
) -> dict[str, Any]:
    """Settings for one run, read from config.json and timing.json when they exist."""
    if config is None:
        fallback = dict(spec.get("fallback_settings", {}))
        source = fallback.pop("source", "sin config.json")
        settings: dict[str, Any] = {
            "algorithm": leaderboard.get("algorithm"),
            "observation": fallback.get("observation"),
            "network": fallback.get("network"),
            "buffer_size": fallback.get("buffer_size"),
            "replay": None,
            "reward_mode": fallback.get("reward_mode"),
            "reward_scale": None,
            "life_loss_terminal": False,
            "seed": fallback.get("seed"),
            "total_steps": fallback.get("total_steps"),
            "resumed_from_step": None,
            "device": None,
            "steps_per_second": None,
            "source": source,
        }
    else:
        cfg: dict[str, Any] = config.get("config", {})
        is_pixels = config.get("observation") == "pixels"
        if "reward_mode" in cfg:
            reward_mode = cfg["reward_mode"]
        elif "reward_clip" in cfg:
            reward_mode = "sign" if cfg["reward_clip"] else "raw"
        else:
            reward_mode = None
        settings = {
            "algorithm": leaderboard.get("algorithm") or config.get("algorithm"),
            "observation": "píxeles, 4 cuadros de 84×84" if is_pixels else "RAM (128 bytes)",
            "network": "CNN (DQN de la Nature)" if is_pixels else "MLP 128-256-256-5",
            "buffer_size": cfg.get("buffer_size"),
            "replay": cfg.get("replay_impl"),
            "reward_mode": reward_mode,
            "reward_scale": cfg.get("reward_scale") if reward_mode == "scale" else None,
            "life_loss_terminal": bool(cfg.get("life_loss_terminal", False)),
            "seed": cfg.get("seed"),
            "total_steps": cfg.get("total_steps"),
            "resumed_from_step": config.get("resumed_from_step"),
            "device": cfg.get("device") or config.get("device"),
            "steps_per_second": None,
            "source": "config.json",
        }
        if settings["replay"] is None and is_pixels:
            # Configs written before H9 have no replay_impl key; they used the stacked layout.
            settings["replay_inferred"] = True
            settings["replay"] = "stacked"
    if timing is not None:
        settings["steps_per_second"] = timing.get("steps_per_second")
        settings["device"] = settings["device"] or timing.get("device")
        if settings["resumed_from_step"] is None:
            settings["resumed_from_step"] = timing.get("resumed_from_step")
    return settings


def _copy_asset(source: Path, target: Path, notes: list[str], rel: str) -> bool:
    """Copy one asset; return False and add a note when the source file is missing."""
    if not source.is_file():
        notes.append(f"Falta el archivo {rel}")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def _build_run(
    run_dir: Path, spec: dict[str, Any], out_assets: Path, page_assets: str,
) -> dict[str, Any]:
    """Build the data for one run folder and copy its referenced assets."""
    leaderboard = _read_json(run_dir / "leaderboard.json") or {}
    config = _read_json(run_dir / "config.json")
    timing = _read_json(run_dir / "timing.json")
    notes: list[str] = []
    if spec.get("partial"):
        notes.append(spec["partial"])
    run_id = run_dir.name
    run_assets = out_assets / run_id
    rows: list[dict[str, Any]] = []
    for raw in leaderboard.get("rows", []):
        screenshots: dict[str, str] = {}
        for name, rel in sorted(raw.get("screenshots", {}).items()):
            if _copy_asset(run_dir / rel, run_assets / rel, notes, f"{run_id}/{rel}"):
                screenshots[name] = f"{page_assets}/{run_id}/{rel}"
        gif_rel = raw.get("gif")
        gif = None
        if gif_rel and _copy_asset(run_dir / gif_rel, run_assets / gif_rel, notes, f"{run_id}/{gif_rel}"):
            gif = f"{page_assets}/{run_id}/{gif_rel}"
        rows.append(
            {
                "title": _title_for(raw),
                "step": raw.get("step"),
                "episodes": raw.get("episodes"),
                "score_mean": raw.get("score_mean"),
                "score_std": raw.get("score_std"),
                "max_rows_mean": raw.get("max_rows_mean"),
                "max_rows_max": raw.get("max_rows_max"),
                "homes_mean": raw.get("homes_mean"),
                "homes_max": raw.get("homes_max"),
                "medals": {
                    name: float(raw.get("medals", {}).get(name, 0.0)) for name in MEDAL_ORDER
                },
                "gif": gif,
                "screenshots": screenshots,
            }
        )
    settings = _settings(spec, config, timing, leaderboard)
    if config is None:
        notes.append("Sin config.json: la configuración viene del README")
    if timing is None:
        notes.append("Sin timing.json: no hay pasos por segundo registrados")
    steps = [row["step"] for row in rows if row["step"] is not None]
    last_step = max(steps) if steps else None
    if config is not None and last_step is not None:
        target = (config.get("config", {}) or {}).get("total_steps")
        if target is not None and last_step < target and not spec.get("partial"):
            notes.append(f"El último checkpoint de la tabla es {last_step:,} de {target:,} pasos".replace(",", " "))
    return {
        "id": run_id,
        "label": spec.get("label", run_id),
        "change": spec.get("change", ""),
        "status": "partial" if spec.get("partial") else "complete",
        "notes": notes,
        "settings": settings,
        "rows": rows,
    }


def _best(run: dict[str, Any]) -> dict[str, Any]:
    """Best score and best frog_home share among a run's checkpoints (random excluded)."""
    checkpoints = [row for row in run["rows"] if row["step"] is not None]
    if not checkpoints:
        return {"score_mean": None, "score_title": None, "frog_home": None, "frog_home_title": None}
    by_score = max(checkpoints, key=lambda row: row["score_mean"] or 0.0)
    by_home = max(checkpoints, key=lambda row: row["medals"]["frog_home"])
    return {
        "score_mean": by_score["score_mean"],
        "score_title": by_score["title"],
        "frog_home": by_home["medals"]["frog_home"],
        "frog_home_title": by_home["title"],
    }


def _hero(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the checkpoint with the highest frog_home share; ties go to the higher score.

    The random row comes from the same run, so both GIFs share one setup.
    """
    best_run: dict[str, Any] | None = None
    best_row: dict[str, Any] | None = None
    best_key: tuple[float, float] = (-1.0, -1.0)
    for run in runs:
        for row in run["rows"]:
            if row["step"] is None or row["gif"] is None:
                continue
            key = (row["medals"]["frog_home"], row["score_mean"] or 0.0)
            if key > best_key:
                best_key, best_run, best_row = key, run, row
    if best_run is None or best_row is None:
        return None
    random_row = next((row for row in best_run["rows"] if row["step"] is None), None)
    return {
        "run_id": best_run["id"],
        "run_label": best_run["label"],
        "checkpoint": best_row,
        "random": random_row,
        "reason": "la mayor proporción de episodios con casa; en empate, el mayor puntaje",
    }


def _benchmark(results_dir: Path) -> dict[str, Any] | None:
    """Summarize results/benchmark_cpu_gpu.json for the lessons section."""
    data = _read_json(results_dir / "benchmark_cpu_gpu.json")
    if data is None:
        return None
    return {
        "source": "results/benchmark_cpu_gpu.json",
        "gpu": data.get("gpu"),
        "env_only_steps_per_second": data.get("env_only_steps_per_s"),
        "update_ms": {"cpu": data.get("update_ms_cpu"), "cuda": data.get("update_ms_cuda")},
        "train_steps_per_second": {
            "cpu": (data.get("train_cpu") or {}).get("steps_per_s"),
            "cuda": (data.get("train_cuda") or {}).get("steps_per_s"),
        },
    }


def build(results_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Copy assets and return the story data (the same object written to data.js)."""
    out_assets = out_dir / "assets"
    if out_assets.exists():
        shutil.rmtree(out_assets)
    out_assets.mkdir(parents=True)
    known = {spec["id"]: spec for spec in RUN_SPECS}
    folders = [
        path.parent
        for path in sorted(results_dir.glob("*/leaderboard.json"))
    ]
    ordered = sorted(
        folders,
        key=lambda folder: (
            0 if folder.name in known else 1,
            list(known).index(folder.name) if folder.name in known else 0,
            folder.name,
        ),
    )
    runs = [
        _build_run(folder, known.get(folder.name, {"id": folder.name}), out_assets, "assets")
        for folder in ordered
    ]
    return {
        "runs": runs,
        "best_by_run": {run["id"]: _best(run) for run in runs},
        "hero": _hero(runs),
        "home_detection": HOME_DETECTION,
        "medal_ladder": MEDAL_LADDER,
        "pilot_h2": PILOT_H2,
        "memory": MEMORY_NOTES,
        "benchmark": _benchmark(results_dir),
    }


def write_data_js(data: dict[str, Any], out_dir: Path) -> Path:
    """Write ``window.STORY_DATA = {...};`` to ``out_dir/data.js`` and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "data.js"
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    target.write_text(f"window.STORY_DATA = {payload};\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.results.is_dir():
        print(f"No existe la carpeta de resultados: {args.results}", file=sys.stderr)
        return 1
    data = build(args.results, args.out)
    target = write_data_js(data, args.out)
    missing = [note for run in data["runs"] for note in run["notes"] if note.startswith("Falta")]
    for note in missing:
        print(f"advertencia: {note}", file=sys.stderr)
    print(f"{len(data['runs'])} corridas escritas en {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
