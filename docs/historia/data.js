window.STORY_DATA = {
  "benchmark": {
    "env_only_steps_per_second": 2191.4,
    "gpu": "NVIDIA GeForce GTX 1660 SUPER",
    "source": "results/benchmark_cpu_gpu.json",
    "train_steps_per_second": {
      "cpu": 751.8,
      "cuda": 580.6
    },
    "update_ms": {
      "cpu": 2.135,
      "cuda": 3.371
    }
  },
  "best_by_run": {
    "run_2m": {
      "frog_home": 0.0,
      "frog_home_title": "100 000",
      "score_mean": 39.3,
      "score_title": "2 000 000"
    },
    "run_px_5m": {
      "frog_home": 0.05,
      "frog_home_title": "1 000 000",
      "score_mean": 44.25,
      "score_title": "3 000 000"
    },
    "run_px_5m_r3": {
      "frog_home": 0.6,
      "frog_home_title": "5 000 000",
      "score_mean": 78.15,
      "score_title": "5 000 000"
    },
    "run_px_5m_r3_seed43_mac": {
      "frog_home": 0.0,
      "frog_home_title": "100 000",
      "score_mean": 36.4,
      "score_title": "5 000 000"
    },
    "run_px_5m_r4": {
      "frog_home": 0.9,
      "frog_home_title": "5 000 000",
      "score_mean": 101.2,
      "score_title": "5 000 000"
    },
    "run_px_5m_r5": {
      "frog_home": 0.95,
      "frog_home_title": "3 000 000",
      "score_mean": 122.15,
      "score_title": "5 000 000"
    },
    "run_px_5m_safe": {
      "frog_home": 0.0,
      "frog_home_title": "100 000",
      "score_mean": 38.15,
      "score_title": "5 000 000"
    }
  },
  "hero": {
    "checkpoint": {
      "episodes": 20,
      "gif": "assets/run_px_5m_r5/gifs/step_3000000.gif",
      "homes_max": 2,
      "homes_mean": 1.0,
      "max_rows_max": 12,
      "max_rows_mean": 11.95,
      "medals": {
        "first_hop": 1.0,
        "first_log": 1.0,
        "frog_home": 0.95,
        "level_cleared": 0.0,
        "mid_river": 1.0,
        "road_crossed": 1.0,
        "three_homes": 0.0
      },
      "score_mean": 101.9,
      "score_std": 18.487563387315266,
      "screenshots": {
        "first_hop": "assets/run_px_5m_r5/screenshots/step_3000000_first_hop.png",
        "first_log": "assets/run_px_5m_r5/screenshots/step_3000000_first_log.png",
        "frog_home": "assets/run_px_5m_r5/screenshots/step_3000000_frog_home.png",
        "mid_river": "assets/run_px_5m_r5/screenshots/step_3000000_mid_river.png",
        "road_crossed": "assets/run_px_5m_r5/screenshots/step_3000000_road_crossed.png"
      },
      "step": 3000000,
      "title": "3 000 000"
    },
    "random": {
      "episodes": 20,
      "gif": "assets/run_px_5m_r5/gifs/random.gif",
      "homes_max": 0,
      "homes_mean": 0.0,
      "max_rows_max": 5,
      "max_rows_mean": 3.0,
      "medals": {
        "first_hop": 1.0,
        "first_log": 0.0,
        "frog_home": 0.0,
        "level_cleared": 0.0,
        "mid_river": 0.0,
        "road_crossed": 0.0,
        "three_homes": 0.0
      },
      "score_mean": 9.3,
      "score_std": 1.6763054614240214,
      "screenshots": {},
      "step": null,
      "title": "Aleatoria"
    },
    "reason": "la mayor proporción de episodios con casa; en empate, el mayor puntaje",
    "run_id": "run_px_5m_r5",
    "run_label": "Corrida 5"
  },
  "home_detection": {
    "evidence": "Verificada con una traza paso a paso (checkpoint de 1 M, semilla 900008): al llegar a casa desde la fila 11, la recompensa es +55, las vidas siguen en 3 y la fila vuelve a 0.",
    "status": "verified"
  },
  "medal_ladder": [
    {
      "emoji": "🐸",
      "key": "first_hop",
      "name": "Primer salto",
      "threshold": "fila 1"
    },
    {
      "emoji": "🥉",
      "key": "road_crossed",
      "name": "Cruce de la carretera",
      "threshold": "fila 6"
    },
    {
      "emoji": "🪵",
      "key": "first_log",
      "name": "Primer tronco",
      "threshold": "fila 7"
    },
    {
      "emoji": "🥈",
      "key": "mid_river",
      "name": "Mitad del río",
      "threshold": "fila 9"
    },
    {
      "emoji": "🥇",
      "key": "frog_home",
      "name": "Casa",
      "threshold": "1 casa en el episodio"
    },
    {
      "emoji": "🏆",
      "key": "three_homes",
      "name": "Tres casas",
      "threshold": "3 casas en el episodio"
    },
    {
      "emoji": "👑",
      "key": "level_cleared",
      "name": "Nivel completo",
      "threshold": "5 casas en el episodio"
    }
  ],
  "memory": {
    "buffer_size": 200000,
    "frame_gb": 1.42,
    "source": "docs/notas-de-desarrollo.md, sección 7",
    "stacked_gb": 11.3
  },
  "pilot_h2": {
    "all_cpu_steps_per_second": 122.8,
    "gpu_util_mean_percent": 45.6,
    "mixed_steps_per_second": 607.0,
    "source": "odd/tasks/home-challenge.md, H2 (100k pasos, 8 entornos)",
    "steps": 100000
  },
  "runs": [
    {
      "change": "Observación RAM de 128 bytes con una MLP. Es la entrega del taller, sin cambios.",
      "id": "run_2m",
      "label": "RAM + MLP (entrega)",
      "notes": [
        "Sin config.json: la configuración viene del README"
      ],
      "rows": [
        {
          "episodes": 20,
          "gif": "assets/run_2m/gifs/random.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 7,
          "max_rows_mean": 3.3,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.05,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.05,
            "three_homes": 0.0
          },
          "score_mean": 9.4,
          "score_std": 2.2891046284519194,
          "screenshots": {},
          "step": null,
          "title": "Aleatoria"
        },
        {
          "episodes": 20,
          "gif": "assets/run_2m/gifs/step_100000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 7,
          "max_rows_mean": 3.5,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.05,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.05,
            "three_homes": 0.0
          },
          "score_mean": 12.15,
          "score_std": 1.904599695474091,
          "screenshots": {
            "first_hop": "assets/run_2m/screenshots/step_100000_first_hop.png",
            "first_log": "assets/run_2m/screenshots/step_100000_first_log.png",
            "road_crossed": "assets/run_2m/screenshots/step_100000_road_crossed.png"
          },
          "step": 100000,
          "title": "100 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_2m/gifs/step_250000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 7.65,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.95,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.2,
            "road_crossed": 0.95,
            "three_homes": 0.0
          },
          "score_mean": 20.6,
          "score_std": 3.2310988842807027,
          "screenshots": {
            "first_hop": "assets/run_2m/screenshots/step_250000_first_hop.png",
            "first_log": "assets/run_2m/screenshots/step_250000_first_log.png",
            "mid_river": "assets/run_2m/screenshots/step_250000_mid_river.png",
            "road_crossed": "assets/run_2m/screenshots/step_250000_road_crossed.png"
          },
          "step": 250000,
          "title": "250 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_2m/gifs/step_500000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 9,
          "max_rows_mean": 8.5,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.9,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.9,
            "road_crossed": 0.9,
            "three_homes": 0.0
          },
          "score_mean": 18.0,
          "score_std": 2.04939015319192,
          "screenshots": {
            "first_hop": "assets/run_2m/screenshots/step_500000_first_hop.png",
            "first_log": "assets/run_2m/screenshots/step_500000_first_log.png",
            "mid_river": "assets/run_2m/screenshots/step_500000_mid_river.png",
            "road_crossed": "assets/run_2m/screenshots/step_500000_road_crossed.png"
          },
          "step": 500000,
          "title": "500 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_2m/gifs/step_1000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.7,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.95,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 32.0,
          "score_std": 4.277849927241488,
          "screenshots": {
            "first_hop": "assets/run_2m/screenshots/step_1000000_first_hop.png",
            "first_log": "assets/run_2m/screenshots/step_1000000_first_log.png",
            "mid_river": "assets/run_2m/screenshots/step_1000000_mid_river.png",
            "road_crossed": "assets/run_2m/screenshots/step_1000000_road_crossed.png"
          },
          "step": 1000000,
          "title": "1 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_2m/gifs/step_2000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.9,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 39.3,
          "score_std": 2.9171904291629644,
          "screenshots": {
            "first_hop": "assets/run_2m/screenshots/step_2000000_first_hop.png",
            "first_log": "assets/run_2m/screenshots/step_2000000_first_log.png",
            "mid_river": "assets/run_2m/screenshots/step_2000000_mid_river.png",
            "road_crossed": "assets/run_2m/screenshots/step_2000000_road_crossed.png"
          },
          "step": 2000000,
          "title": "2 000 000"
        }
      ],
      "settings": {
        "algorithm": "DQN",
        "buffer_size": 100000,
        "device": "cpu",
        "life_loss_terminal": false,
        "network": "MLP 128-256-256-5",
        "observation": "RAM (128 bytes)",
        "replay": null,
        "resumed_from_step": null,
        "reward_mode": "sign",
        "reward_scale": null,
        "seed": 42,
        "source": "README.md (la corrida no tiene config.json)",
        "steps_per_second": 738.9,
        "total_steps": 2000000
      },
      "status": "complete"
    },
    {
      "change": "Píxeles y CNN, buffer de 200 000, recompensa con signo.",
      "id": "run_px_5m",
      "label": "Corrida 1",
      "notes": [
        "Detenida por falta de memoria en 4,6 M de pasos (episodes.csv). Los checkpoints llegan hasta 3 M.",
        "Sin timing.json: no hay pasos por segundo registrados"
      ],
      "rows": [
        {
          "episodes": 20,
          "gif": "assets/run_px_5m/gifs/random.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 3.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 9.3,
          "score_std": 1.6763054614240214,
          "screenshots": {},
          "step": null,
          "title": "Aleatoria"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m/gifs/step_100000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 3.3,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 8.7,
          "score_std": 1.6462077633154328,
          "screenshots": {
            "first_hop": "assets/run_px_5m/screenshots/step_100000_first_hop.png"
          },
          "step": 100000,
          "title": "100 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m/gifs/step_250000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 8,
          "max_rows_mean": 4.9,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.3,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.35,
            "three_homes": 0.0
          },
          "score_mean": 14.1,
          "score_std": 2.1656407827707715,
          "screenshots": {
            "first_hop": "assets/run_px_5m/screenshots/step_250000_first_hop.png",
            "first_log": "assets/run_px_5m/screenshots/step_250000_first_log.png",
            "road_crossed": "assets/run_px_5m/screenshots/step_250000_road_crossed.png"
          },
          "step": 250000,
          "title": "250 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m/gifs/step_500000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 8,
          "max_rows_mean": 6.2,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.5,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.75,
            "three_homes": 0.0
          },
          "score_mean": 19.55,
          "score_std": 3.6121323342314025,
          "screenshots": {
            "first_hop": "assets/run_px_5m/screenshots/step_500000_first_hop.png",
            "first_log": "assets/run_px_5m/screenshots/step_500000_first_log.png",
            "road_crossed": "assets/run_px_5m/screenshots/step_500000_road_crossed.png"
          },
          "step": 500000,
          "title": "500 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m/gifs/step_1000000.gif",
          "homes_max": 1,
          "homes_mean": 0.05,
          "max_rows_max": 12,
          "max_rows_mean": 9.2,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.05,
            "level_cleared": 0.0,
            "mid_river": 0.85,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 36.6,
          "score_std": 12.920526305069776,
          "screenshots": {
            "first_hop": "assets/run_px_5m/screenshots/step_1000000_first_hop.png",
            "first_log": "assets/run_px_5m/screenshots/step_1000000_first_log.png",
            "frog_home": "assets/run_px_5m/screenshots/step_1000000_frog_home.png",
            "mid_river": "assets/run_px_5m/screenshots/step_1000000_mid_river.png",
            "road_crossed": "assets/run_px_5m/screenshots/step_1000000_road_crossed.png"
          },
          "step": 1000000,
          "title": "1 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m/gifs/step_2000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 11,
          "max_rows_mean": 10.5,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 44.1,
          "score_std": 2.1656407827707715,
          "screenshots": {
            "first_hop": "assets/run_px_5m/screenshots/step_2000000_first_hop.png",
            "first_log": "assets/run_px_5m/screenshots/step_2000000_first_log.png",
            "mid_river": "assets/run_px_5m/screenshots/step_2000000_mid_river.png",
            "road_crossed": "assets/run_px_5m/screenshots/step_2000000_road_crossed.png"
          },
          "step": 2000000,
          "title": "2 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m/gifs/step_3000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 11,
          "max_rows_mean": 10.8,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 44.25,
          "score_std": 1.6085707942145413,
          "screenshots": {
            "first_hop": "assets/run_px_5m/screenshots/step_3000000_first_hop.png",
            "first_log": "assets/run_px_5m/screenshots/step_3000000_first_log.png",
            "mid_river": "assets/run_px_5m/screenshots/step_3000000_mid_river.png",
            "road_crossed": "assets/run_px_5m/screenshots/step_3000000_road_crossed.png"
          },
          "step": 3000000,
          "title": "3 000 000"
        }
      ],
      "settings": {
        "algorithm": "DQN-pixels (Double, n-step)",
        "buffer_size": 200000,
        "device": "cuda",
        "life_loss_terminal": false,
        "network": "CNN (DQN de la Nature)",
        "observation": "píxeles, 4 cuadros de 84×84",
        "replay": "stacked",
        "replay_inferred": true,
        "resumed_from_step": null,
        "reward_mode": "sign",
        "reward_scale": null,
        "seed": 42,
        "source": "config.json",
        "steps_per_second": null,
        "total_steps": 5000000
      },
      "status": "partial"
    },
    {
      "change": "Igual que la corrida 1, con buffer de 100 000.",
      "id": "run_px_5m_safe",
      "label": "Corrida 2",
      "notes": [],
      "rows": [
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_safe/gifs/random.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 3.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 9.3,
          "score_std": 1.6763054614240214,
          "screenshots": {},
          "step": null,
          "title": "Aleatoria"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_safe/gifs/step_100000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 7,
          "max_rows_mean": 3.95,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.05,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.05,
            "three_homes": 0.0
          },
          "score_mean": 10.8,
          "score_std": 2.0396078054371136,
          "screenshots": {
            "first_hop": "assets/run_px_5m_safe/screenshots/step_100000_first_hop.png",
            "first_log": "assets/run_px_5m_safe/screenshots/step_100000_first_log.png",
            "road_crossed": "assets/run_px_5m_safe/screenshots/step_100000_road_crossed.png"
          },
          "step": 100000,
          "title": "100 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_safe/gifs/step_250000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 7,
          "max_rows_mean": 5.25,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.3,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.45,
            "three_homes": 0.0
          },
          "score_mean": 13.35,
          "score_std": 1.5580436450882884,
          "screenshots": {
            "first_hop": "assets/run_px_5m_safe/screenshots/step_250000_first_hop.png",
            "first_log": "assets/run_px_5m_safe/screenshots/step_250000_first_log.png",
            "road_crossed": "assets/run_px_5m_safe/screenshots/step_250000_road_crossed.png"
          },
          "step": 250000,
          "title": "250 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_safe/gifs/step_500000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 9,
          "max_rows_mean": 7.05,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.7,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.05,
            "road_crossed": 0.75,
            "three_homes": 0.0
          },
          "score_mean": 22.1,
          "score_std": 5.3282267219028885,
          "screenshots": {
            "first_hop": "assets/run_px_5m_safe/screenshots/step_500000_first_hop.png",
            "first_log": "assets/run_px_5m_safe/screenshots/step_500000_first_log.png",
            "mid_river": "assets/run_px_5m_safe/screenshots/step_500000_mid_river.png",
            "road_crossed": "assets/run_px_5m_safe/screenshots/step_500000_road_crossed.png"
          },
          "step": 500000,
          "title": "500 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_safe/gifs/step_1000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.05,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.85,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 37.95,
          "score_std": 3.499642838919423,
          "screenshots": {
            "first_hop": "assets/run_px_5m_safe/screenshots/step_1000000_first_hop.png",
            "first_log": "assets/run_px_5m_safe/screenshots/step_1000000_first_log.png",
            "mid_river": "assets/run_px_5m_safe/screenshots/step_1000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_safe/screenshots/step_1000000_road_crossed.png"
          },
          "step": 1000000,
          "title": "1 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_safe/gifs/step_2000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.85,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 36.8,
          "score_std": 2.4413111231467406,
          "screenshots": {
            "first_hop": "assets/run_px_5m_safe/screenshots/step_2000000_first_hop.png",
            "first_log": "assets/run_px_5m_safe/screenshots/step_2000000_first_log.png",
            "mid_river": "assets/run_px_5m_safe/screenshots/step_2000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_safe/screenshots/step_2000000_road_crossed.png"
          },
          "step": 2000000,
          "title": "2 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_safe/gifs/step_3000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.8,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 38.0,
          "score_std": 1.61245154965971,
          "screenshots": {
            "first_hop": "assets/run_px_5m_safe/screenshots/step_3000000_first_hop.png",
            "first_log": "assets/run_px_5m_safe/screenshots/step_3000000_first_log.png",
            "mid_river": "assets/run_px_5m_safe/screenshots/step_3000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_safe/screenshots/step_3000000_road_crossed.png"
          },
          "step": 3000000,
          "title": "3 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_safe/gifs/step_5000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.65,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 38.15,
          "score_std": 4.077683165720456,
          "screenshots": {
            "first_hop": "assets/run_px_5m_safe/screenshots/step_5000000_first_hop.png",
            "first_log": "assets/run_px_5m_safe/screenshots/step_5000000_first_log.png",
            "mid_river": "assets/run_px_5m_safe/screenshots/step_5000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_safe/screenshots/step_5000000_road_crossed.png"
          },
          "step": 5000000,
          "title": "5 000 000"
        }
      ],
      "settings": {
        "algorithm": "DQN-pixels (Double, n-step)",
        "buffer_size": 100000,
        "device": "cuda",
        "life_loss_terminal": false,
        "network": "CNN (DQN de la Nature)",
        "observation": "píxeles, 4 cuadros de 84×84",
        "replay": "stacked",
        "replay_inferred": true,
        "resumed_from_step": null,
        "reward_mode": "sign",
        "reward_scale": null,
        "seed": 42,
        "source": "config.json",
        "steps_per_second": 555.1,
        "total_steps": 5000000
      },
      "status": "complete"
    },
    {
      "change": "Buffer de 200 000 con cuadros compartidos, recompensa escalada por 0,1, semilla 42.",
      "id": "run_px_5m_r3",
      "label": "Corrida 3",
      "notes": [
        "Detenida por falta de memoria en 2,74 M de pasos; reanudada desde el checkpoint de 2,5 M."
      ],
      "rows": [
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3/gifs/random.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 3.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 9.3,
          "score_std": 1.6763054614240214,
          "screenshots": {},
          "step": null,
          "title": "Aleatoria"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3/gifs/step_100000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 2.4,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 6.95,
          "score_std": 1.6874537030686205,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3/screenshots/step_100000_first_hop.png"
          },
          "step": 100000,
          "title": "100 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3/gifs/step_250000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 2.65,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 8.15,
          "score_std": 1.8781639970992947,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3/screenshots/step_250000_first_hop.png"
          },
          "step": 250000,
          "title": "250 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3/gifs/step_500000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 8,
          "max_rows_mean": 4.2,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.05,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.05,
            "three_homes": 0.0
          },
          "score_mean": 13.3,
          "score_std": 2.17025344142107,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3/screenshots/step_500000_first_hop.png",
            "first_log": "assets/run_px_5m_r3/screenshots/step_500000_first_log.png",
            "road_crossed": "assets/run_px_5m_r3/screenshots/step_500000_road_crossed.png"
          },
          "step": 500000,
          "title": "500 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3/gifs/step_1000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 7.85,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.95,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.2,
            "road_crossed": 0.95,
            "three_homes": 0.0
          },
          "score_mean": 26.35,
          "score_std": 4.304358256465184,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3/screenshots/step_1000000_first_hop.png",
            "first_log": "assets/run_px_5m_r3/screenshots/step_1000000_first_log.png",
            "mid_river": "assets/run_px_5m_r3/screenshots/step_1000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r3/screenshots/step_1000000_road_crossed.png"
          },
          "step": 1000000,
          "title": "1 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3/gifs/step_2000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 9,
          "max_rows_mean": 9.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 39.7,
          "score_std": 2.551470164434615,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3/screenshots/step_2000000_first_hop.png",
            "first_log": "assets/run_px_5m_r3/screenshots/step_2000000_first_log.png",
            "mid_river": "assets/run_px_5m_r3/screenshots/step_2000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r3/screenshots/step_2000000_road_crossed.png"
          },
          "step": 2000000,
          "title": "2 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3/gifs/step_3000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.15,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 41.75,
          "score_std": 2.405722344743882,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3/screenshots/step_3000000_first_hop.png",
            "first_log": "assets/run_px_5m_r3/screenshots/step_3000000_first_log.png",
            "mid_river": "assets/run_px_5m_r3/screenshots/step_3000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r3/screenshots/step_3000000_road_crossed.png"
          },
          "step": 3000000,
          "title": "3 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3/gifs/step_5000000.gif",
          "homes_max": 1,
          "homes_mean": 0.6,
          "max_rows_max": 12,
          "max_rows_mean": 11.05,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.6,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 78.15,
          "score_std": 36.80662304531618,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3/screenshots/step_5000000_first_hop.png",
            "first_log": "assets/run_px_5m_r3/screenshots/step_5000000_first_log.png",
            "frog_home": "assets/run_px_5m_r3/screenshots/step_5000000_frog_home.png",
            "mid_river": "assets/run_px_5m_r3/screenshots/step_5000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r3/screenshots/step_5000000_road_crossed.png"
          },
          "step": 5000000,
          "title": "5 000 000"
        }
      ],
      "settings": {
        "algorithm": "DQN-pixels (Double, n-step)",
        "buffer_size": 200000,
        "device": "cuda",
        "life_loss_terminal": false,
        "network": "CNN (DQN de la Nature)",
        "observation": "píxeles, 4 cuadros de 84×84",
        "replay": "frame",
        "resumed_from_step": 2500000,
        "reward_mode": "scale",
        "reward_scale": 0.1,
        "seed": 42,
        "source": "config.json",
        "steps_per_second": 552.7,
        "total_steps": 5000000
      },
      "status": "partial"
    },
    {
      "change": "Buffer de 200 000 con cuadros compartidos y recompensa con signo.",
      "id": "run_px_5m_r4",
      "label": "Corrida 4",
      "notes": [],
      "rows": [
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r4/gifs/random.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 3.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 9.3,
          "score_std": 1.6763054614240214,
          "screenshots": {},
          "step": null,
          "title": "Aleatoria"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r4/gifs/step_100000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 3.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 8.05,
          "score_std": 2.1324868112136124,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r4/screenshots/step_100000_first_hop.png"
          },
          "step": 100000,
          "title": "100 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r4/gifs/step_250000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 7,
          "max_rows_mean": 3.55,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.05,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.05,
            "three_homes": 0.0
          },
          "score_mean": 11.45,
          "score_std": 2.132486811213612,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r4/screenshots/step_250000_first_hop.png",
            "first_log": "assets/run_px_5m_r4/screenshots/step_250000_first_log.png",
            "road_crossed": "assets/run_px_5m_r4/screenshots/step_250000_road_crossed.png"
          },
          "step": 250000,
          "title": "250 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r4/gifs/step_500000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 8.3,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.2,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 27.3,
          "score_std": 3.7296112397943038,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r4/screenshots/step_500000_first_hop.png",
            "first_log": "assets/run_px_5m_r4/screenshots/step_500000_first_log.png",
            "mid_river": "assets/run_px_5m_r4/screenshots/step_500000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r4/screenshots/step_500000_road_crossed.png"
          },
          "step": 500000,
          "title": "500 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r4/gifs/step_1000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.95,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 33.65,
          "score_std": 5.14076842505087,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r4/screenshots/step_1000000_first_hop.png",
            "first_log": "assets/run_px_5m_r4/screenshots/step_1000000_first_log.png",
            "mid_river": "assets/run_px_5m_r4/screenshots/step_1000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r4/screenshots/step_1000000_road_crossed.png"
          },
          "step": 1000000,
          "title": "1 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r4/gifs/step_2000000.gif",
          "homes_max": 1,
          "homes_mean": 0.35,
          "max_rows_max": 12,
          "max_rows_mean": 11.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.35,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 66.45,
          "score_std": 31.504721868316818,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r4/screenshots/step_2000000_first_hop.png",
            "first_log": "assets/run_px_5m_r4/screenshots/step_2000000_first_log.png",
            "frog_home": "assets/run_px_5m_r4/screenshots/step_2000000_frog_home.png",
            "mid_river": "assets/run_px_5m_r4/screenshots/step_2000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r4/screenshots/step_2000000_road_crossed.png"
          },
          "step": 2000000,
          "title": "2 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r4/gifs/step_3000000.gif",
          "homes_max": 1,
          "homes_mean": 0.85,
          "max_rows_max": 12,
          "max_rows_mean": 11.55,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.85,
            "level_cleared": 0.0,
            "mid_river": 0.95,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 97.15,
          "score_std": 25.68321436269222,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r4/screenshots/step_3000000_first_hop.png",
            "first_log": "assets/run_px_5m_r4/screenshots/step_3000000_first_log.png",
            "frog_home": "assets/run_px_5m_r4/screenshots/step_3000000_frog_home.png",
            "mid_river": "assets/run_px_5m_r4/screenshots/step_3000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r4/screenshots/step_3000000_road_crossed.png"
          },
          "step": 3000000,
          "title": "3 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r4/gifs/step_5000000.gif",
          "homes_max": 1,
          "homes_mean": 0.9,
          "max_rows_max": 12,
          "max_rows_mean": 11.85,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.9,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 101.2,
          "score_std": 19.296631830451656,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r4/screenshots/step_5000000_first_hop.png",
            "first_log": "assets/run_px_5m_r4/screenshots/step_5000000_first_log.png",
            "frog_home": "assets/run_px_5m_r4/screenshots/step_5000000_frog_home.png",
            "mid_river": "assets/run_px_5m_r4/screenshots/step_5000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r4/screenshots/step_5000000_road_crossed.png"
          },
          "step": 5000000,
          "title": "5 000 000"
        }
      ],
      "settings": {
        "algorithm": "DQN-pixels (Double, n-step)",
        "buffer_size": 200000,
        "device": "cuda",
        "life_loss_terminal": false,
        "network": "CNN (DQN de la Nature)",
        "observation": "píxeles, 4 cuadros de 84×84",
        "replay": "frame",
        "resumed_from_step": null,
        "reward_mode": "sign",
        "reward_scale": null,
        "seed": 42,
        "source": "config.json",
        "steps_per_second": 550.9,
        "total_steps": 5000000
      },
      "status": "complete"
    },
    {
      "change": "Igual que la corrida 4, con la pérdida de una vida como fin para el aprendizaje.",
      "id": "run_px_5m_r5",
      "label": "Corrida 5",
      "notes": [],
      "rows": [
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r5/gifs/random.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 3.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 9.3,
          "score_std": 1.6763054614240214,
          "screenshots": {},
          "step": null,
          "title": "Aleatoria"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r5/gifs/step_100000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 3.95,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 11.5,
          "score_std": 1.3228756555322954,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r5/screenshots/step_100000_first_hop.png"
          },
          "step": 100000,
          "title": "100 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r5/gifs/step_250000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 8,
          "max_rows_mean": 5.5,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.25,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.45,
            "three_homes": 0.0
          },
          "score_mean": 17.2,
          "score_std": 2.420743687382041,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r5/screenshots/step_250000_first_hop.png",
            "first_log": "assets/run_px_5m_r5/screenshots/step_250000_first_log.png",
            "road_crossed": "assets/run_px_5m_r5/screenshots/step_250000_road_crossed.png"
          },
          "step": 250000,
          "title": "250 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r5/gifs/step_500000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 8.15,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.9,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.4,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 29.45,
          "score_std": 3.8791107228332633,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r5/screenshots/step_500000_first_hop.png",
            "first_log": "assets/run_px_5m_r5/screenshots/step_500000_first_log.png",
            "mid_river": "assets/run_px_5m_r5/screenshots/step_500000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r5/screenshots/step_500000_road_crossed.png"
          },
          "step": 500000,
          "title": "500 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r5/gifs/step_1000000.gif",
          "homes_max": 1,
          "homes_mean": 0.05,
          "max_rows_max": 12,
          "max_rows_mean": 9.85,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.05,
            "level_cleared": 0.0,
            "mid_river": 0.9,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 44.35,
          "score_std": 8.878485231164154,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r5/screenshots/step_1000000_first_hop.png",
            "first_log": "assets/run_px_5m_r5/screenshots/step_1000000_first_log.png",
            "frog_home": "assets/run_px_5m_r5/screenshots/step_1000000_frog_home.png",
            "mid_river": "assets/run_px_5m_r5/screenshots/step_1000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r5/screenshots/step_1000000_road_crossed.png"
          },
          "step": 1000000,
          "title": "1 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r5/gifs/step_2000000.gif",
          "homes_max": 1,
          "homes_mean": 0.4,
          "max_rows_max": 12,
          "max_rows_mean": 11.15,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.4,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 66.35,
          "score_std": 27.93255269394475,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r5/screenshots/step_2000000_first_hop.png",
            "first_log": "assets/run_px_5m_r5/screenshots/step_2000000_first_log.png",
            "frog_home": "assets/run_px_5m_r5/screenshots/step_2000000_frog_home.png",
            "mid_river": "assets/run_px_5m_r5/screenshots/step_2000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r5/screenshots/step_2000000_road_crossed.png"
          },
          "step": 2000000,
          "title": "2 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r5/gifs/step_3000000.gif",
          "homes_max": 2,
          "homes_mean": 1.0,
          "max_rows_max": 12,
          "max_rows_mean": 11.95,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.95,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 101.9,
          "score_std": 18.487563387315266,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r5/screenshots/step_3000000_first_hop.png",
            "first_log": "assets/run_px_5m_r5/screenshots/step_3000000_first_log.png",
            "frog_home": "assets/run_px_5m_r5/screenshots/step_3000000_frog_home.png",
            "mid_river": "assets/run_px_5m_r5/screenshots/step_3000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r5/screenshots/step_3000000_road_crossed.png"
          },
          "step": 3000000,
          "title": "3 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r5/gifs/step_5000000.gif",
          "homes_max": 3,
          "homes_mean": 1.35,
          "max_rows_max": 12,
          "max_rows_mean": 11.8,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.9,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.05
          },
          "score_mean": 122.15,
          "score_std": 50.52848206704808,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r5/screenshots/step_5000000_first_hop.png",
            "first_log": "assets/run_px_5m_r5/screenshots/step_5000000_first_log.png",
            "frog_home": "assets/run_px_5m_r5/screenshots/step_5000000_frog_home.png",
            "mid_river": "assets/run_px_5m_r5/screenshots/step_5000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r5/screenshots/step_5000000_road_crossed.png",
            "three_homes": "assets/run_px_5m_r5/screenshots/step_5000000_three_homes.png"
          },
          "step": 5000000,
          "title": "5 000 000"
        }
      ],
      "settings": {
        "algorithm": "DQN-pixels (Double, n-step)",
        "buffer_size": 200000,
        "device": "cuda",
        "life_loss_terminal": true,
        "network": "CNN (DQN de la Nature)",
        "observation": "píxeles, 4 cuadros de 84×84",
        "replay": "frame",
        "resumed_from_step": null,
        "reward_mode": "sign",
        "reward_scale": null,
        "seed": 42,
        "source": "config.json",
        "steps_per_second": 562.2,
        "total_steps": 5000000
      },
      "status": "complete"
    },
    {
      "change": "Réplica de la corrida 3 con semilla 43, en un MacBook Pro M4 (MPS).",
      "id": "run_px_5m_r3_seed43_mac",
      "label": "Corrida 3b (Mac)",
      "notes": [],
      "rows": [
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3_seed43_mac/gifs/random.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 5,
          "max_rows_mean": 3.0,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 9.3,
          "score_std": 1.6763054614240214,
          "screenshots": {},
          "step": null,
          "title": "Aleatoria"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3_seed43_mac/gifs/step_100000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 3,
          "max_rows_mean": 1.7,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.0,
            "three_homes": 0.0
          },
          "score_mean": 5.75,
          "score_std": 0.7664854858377946,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3_seed43_mac/screenshots/step_100000_first_hop.png"
          },
          "step": 100000,
          "title": "100 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3_seed43_mac/gifs/step_250000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 9,
          "max_rows_mean": 5.15,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.3,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.15,
            "road_crossed": 0.35,
            "three_homes": 0.0
          },
          "score_mean": 13.7,
          "score_std": 3.8742741255621036,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3_seed43_mac/screenshots/step_250000_first_hop.png",
            "first_log": "assets/run_px_5m_r3_seed43_mac/screenshots/step_250000_first_log.png",
            "mid_river": "assets/run_px_5m_r3_seed43_mac/screenshots/step_250000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r3_seed43_mac/screenshots/step_250000_road_crossed.png"
          },
          "step": 250000,
          "title": "250 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3_seed43_mac/gifs/step_500000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 7,
          "max_rows_mean": 3.85,
          "medals": {
            "first_hop": 1.0,
            "first_log": 0.1,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.0,
            "road_crossed": 0.1,
            "three_homes": 0.0
          },
          "score_mean": 12.2,
          "score_std": 2.3579652245103193,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3_seed43_mac/screenshots/step_500000_first_hop.png",
            "first_log": "assets/run_px_5m_r3_seed43_mac/screenshots/step_500000_first_log.png",
            "road_crossed": "assets/run_px_5m_r3_seed43_mac/screenshots/step_500000_road_crossed.png"
          },
          "step": 500000,
          "title": "500 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3_seed43_mac/gifs/step_1000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 9,
          "max_rows_mean": 7.6,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.15,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 25.85,
          "score_std": 3.320767983464066,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3_seed43_mac/screenshots/step_1000000_first_hop.png",
            "first_log": "assets/run_px_5m_r3_seed43_mac/screenshots/step_1000000_first_log.png",
            "mid_river": "assets/run_px_5m_r3_seed43_mac/screenshots/step_1000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r3_seed43_mac/screenshots/step_1000000_road_crossed.png"
          },
          "step": 1000000,
          "title": "1 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3_seed43_mac/gifs/step_2000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 8.55,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 0.45,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 34.0,
          "score_std": 1.5491933384829668,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3_seed43_mac/screenshots/step_2000000_first_hop.png",
            "first_log": "assets/run_px_5m_r3_seed43_mac/screenshots/step_2000000_first_log.png",
            "mid_river": "assets/run_px_5m_r3_seed43_mac/screenshots/step_2000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r3_seed43_mac/screenshots/step_2000000_road_crossed.png"
          },
          "step": 2000000,
          "title": "2 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3_seed43_mac/gifs/step_3000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.85,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 36.0,
          "score_std": 1.5811388300841898,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3_seed43_mac/screenshots/step_3000000_first_hop.png",
            "first_log": "assets/run_px_5m_r3_seed43_mac/screenshots/step_3000000_first_log.png",
            "mid_river": "assets/run_px_5m_r3_seed43_mac/screenshots/step_3000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r3_seed43_mac/screenshots/step_3000000_road_crossed.png"
          },
          "step": 3000000,
          "title": "3 000 000"
        },
        {
          "episodes": 20,
          "gif": "assets/run_px_5m_r3_seed43_mac/gifs/step_5000000.gif",
          "homes_max": 0,
          "homes_mean": 0.0,
          "max_rows_max": 10,
          "max_rows_mean": 9.7,
          "medals": {
            "first_hop": 1.0,
            "first_log": 1.0,
            "frog_home": 0.0,
            "level_cleared": 0.0,
            "mid_river": 1.0,
            "road_crossed": 1.0,
            "three_homes": 0.0
          },
          "score_mean": 36.4,
          "score_std": 0.9165151389911679,
          "screenshots": {
            "first_hop": "assets/run_px_5m_r3_seed43_mac/screenshots/step_5000000_first_hop.png",
            "first_log": "assets/run_px_5m_r3_seed43_mac/screenshots/step_5000000_first_log.png",
            "mid_river": "assets/run_px_5m_r3_seed43_mac/screenshots/step_5000000_mid_river.png",
            "road_crossed": "assets/run_px_5m_r3_seed43_mac/screenshots/step_5000000_road_crossed.png"
          },
          "step": 5000000,
          "title": "5 000 000"
        }
      ],
      "settings": {
        "algorithm": "DQN-pixels (Double, n-step)",
        "buffer_size": 200000,
        "device": "mps",
        "life_loss_terminal": false,
        "network": "CNN (DQN de la Nature)",
        "observation": "píxeles, 4 cuadros de 84×84",
        "replay": "frame",
        "resumed_from_step": null,
        "reward_mode": "scale",
        "reward_scale": 0.1,
        "seed": 43,
        "source": "config.json",
        "steps_per_second": 379.5,
        "total_steps": 5000000
      },
      "status": "complete"
    }
  ]
};
