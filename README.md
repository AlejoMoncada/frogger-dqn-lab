# frogger-dqn-lab

**Taller 2 (U3) — Grupo 2.** Agente DQN implementado desde cero con **PyTorch + Gymnasium**, sin
Stable-Baselines3, que aprende a jugar `ALE/Frogger-v5` a partir de la observación RAM de la consola.

Este repositorio es independiente del taller de PPO (`frogger-rl-lab`), que usa Stable-Baselines3 y se
entregó por separado.

<table>
<tr>
<td align="center"><b>Política aleatoria</b><br><img src="results/run_2m/gifs/random.gif" width="300" alt="Política aleatoria"></td>
<td align="center"><b>DQN tras 2 000 000 de pasos</b><br><img src="results/run_2m/gifs/step_2000000.gif" width="300" alt="DQN a 2M pasos"></td>
</tr>
</table>

## Resumen

| Elemento | Valor |
|---|---|
| Entorno | `ALE/Frogger-v5`, observación RAM (128 bytes), acciones discretas (5) |
| Algoritmo | DQN: buffer de repetición, ε-greedy, red objetivo, pérdida Huber |
| Red | MLP 128 → 256 → 256 → 5, ReLU, 100 101 parámetros |
| Entrenamiento | 2 000 000 de pasos en CPU, 45 min (738.9 pasos/s) |
| Resultado principal | Puntaje greedy 39.3 ± 2.9 a 2M pasos, frente a 9.4 ± 2.3 de la política aleatoria |

## Entorno

### Observación

- **Espacio de observación:** `Box(0, 1, (128,), float32)`. El entorno entrega la RAM de la consola
  (128 bytes, `uint8` en `[0, 255]`), que el wrapper `RamFloatObservation` normaliza dividiendo entre 255
  (`dqn_frogger/env.py`).
- **Por qué RAM y no píxeles:** el DQN clásico usa frames de 84×84 apilados y una CNN, lo que requiere
  millones de pasos. Con el plazo disponible elegimos la RAM: el juego es el mismo y solo cambia la
  representación del estado. El costo es que la RAM no tiene estructura espacial explícita, así que la
  red debe descubrir qué posiciones codifican al sapo y a los autos. Véase
  [`docs/notas-de-desarrollo.md`](docs/notas-de-desarrollo.md), sección 2.

### Acciones

Espacio de acciones `Discrete(5)`:

| Índice | Acción |
|---|---|
| 0 | NOOP |
| 1 | UP |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | DOWN |

Configuración por defecto de la versión `v5` de ALE: `frameskip = 4` y `repeat_action_probability = 0.25`
(sticky actions). El límite de truncamiento es `max_num_frames_per_episode = 108 000` cuadros, es decir,
27 000 pasos del agente con `frameskip = 4`. Todo verificado con `env.spec.kwargs` y el entorno instanciado.

### Recompensas y terminación

- **Recompensa:** +1 por cada salto hacia adelante que alcanza una fila nueva dentro de la vida actual.
  No observamos otros valores de recompensa. La fila se confirmó en RAM: el byte 90 cuenta las filas de la
  vida actual y coincide con el conteo de recompensas en todos los pasos de cinco episodios
  (`docs/notas-de-desarrollo.md`, sección 4).
- **Recorte de recompensa:** el entrenamiento usa el signo de la recompensa (`reward_clip = True`). Como
  todas las recompensas son 0 o 1, el recorte no cambia ningún valor. Los registros (`episodes.csv`) guardan
  el puntaje sin recortar.
- **Vidas:** 4 al inicio (`info["lives"]`).
- **Terminación (`terminated`):** el episodio termina cuando el juego llega a *game over*, es decir, cuando
  las vidas llegan a 0. Verificado: en episodios aleatorios, `terminated = True` coincide con `lives = 0`.
- **Truncamiento (`truncated`):** el entorno corta el episodio al llegar al límite de 108 000 cuadros.
- **Tratamiento en Bellman:** solo `terminated` anula el valor del estado siguiente. Si un episodio se
  trunca, la red sigue estimando el valor futuro. El buffer guarda `done = terminated`
  (`dqn_frogger/trainer.py`).

## Flujo de entrenamiento

<p align="center"><img src="docs/img/training_flow.png" width="520" alt="Diagrama del flujo de entrenamiento del DQN"></p>

<details>
<summary>Código fuente del diagrama (Mermaid)</summary>

```mermaid
flowchart TD
    A["Entorno ALE/Frogger-v5 (RAM normalizada, 128 valores)"] --> B["Acción ε-greedy sobre la red en línea"]
    B --> C["env.step(a) devuelve s', r, terminated, truncated"]
    C --> D["Replay buffer: guardar (s, a, r, s', done)"]
    D --> E{"step >= 10 000 y step mod 4 = 0?"}
    E -- si --> F["Muestra uniforme de 64 transiciones"]
    E -- no --> G{"step mod 2 000 = 0?"}
    F --> H["Q(s, a) con la red en línea"]
    F --> I["Objetivo y = r + gamma (1 - done) max Q_objetivo(s', a')"]
    H --> J["Pérdida Huber entre Q(s, a) e y, Adam y clip de norma 10"]
    I --> J
    J --> G
    G -- si --> K["Copia dura de pesos a la red objetivo"]
    G -- no --> A
    K --> A
```

</details>

### Componentes

- **Replay buffer:** capacidad de 100 000 transiciones en un arreglo circular. Cuando está lleno,
  sobrescribe la más antigua. Cada lote se muestrea uniformemente con reemplazo.
- **ε-greedy:** con probabilidad ε la acción es aleatoria; si no, es la de mayor Q. ε decrece linealmente
  de 1.0 a 0.05 en 150 000 pasos y después se mantiene en 0.05.
- **Red objetivo:** una copia de la red en línea que se actualiza por copia dura cada 2 000 pasos. Estabiliza
  el objetivo de Bellman, porque evita que el objetivo cambie con cada paso de gradiente.
- **Actualización de Bellman:** la red en línea se ajusta hacia el objetivo

  $$
  y = r + \gamma \,(1 - \mathrm{done})\, \max_{a'} Q_{\bar\theta}(s', a')
  $$

  minimizando la pérdida de Huber $\ell_\delta\big(Q_\theta(s, a) - y\big)$ con $\delta = 1$
  (`smooth_l1_loss`). La red objetivo $Q_{\bar\theta}$ no recibe gradiente.
- **Optimización:** Adam con tasa de 1e-4 y recorte de la norma del gradiente en 10.0.

### Hiperparámetros

Valores tomados de `dqn_frogger/config.py` y `dqn_frogger/agent.py`. La corrida de 2M se lanzó con
`--total-steps 2000000`; el resto de valores es el de `config.py` (véase la nota de reproducción sobre `config.json`).

| Parámetro | Valor |
|---|---|
| Pasos totales | 2 000 000 (por defecto en `config.py`: 300 000) |
| Capacidad del buffer | 100 000 |
| Tamaño de lote | 64 |
| Factor de descuento γ | 0.99 |
| Tasa de aprendizaje (Adam) | 1e-4 |
| Inicio del aprendizaje (`learning_starts`) | 10 000 pasos |
| Frecuencia de entrenamiento (`train_freq`) | cada 4 pasos |
| Sincronización de la red objetivo | cada 2 000 pasos |
| ε inicial / final | 1.0 / 0.05 |
| Pasos de decaimiento de ε | 150 000 (lineal) |
| Recorte de recompensa | signo (`reward_clip = True`) |
| Clip de la norma del gradiente | 10.0 |
| Pérdida | Huber (`smooth_l1_loss`, δ = 1) |
| Semilla de entrenamiento | 42 |
| Evaluación durante el entrenamiento | cada 25 000 pasos, 10 episodios, ε = 0.01 |
| Checkpoints | 100 000, 250 000, 500 000, 1 000 000, 2 000 000 (y el paso final) |

## Arquitectura de la red

`QNetwork` (`dqn_frogger/network.py`): una MLP con activaciones ReLU y salida lineal.

```
Entrada (128) → Linear(128, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 5)
```

Número de parámetros: **100 101** (33 024 + 65 792 + 1 285), verificado con `sum(p.numel())`. La red
objetivo tiene la misma arquitectura y el mismo número de parámetros.

## Resultados

## Alcance de la evaluación

Las evaluaciones periódicas y por checkpoints utilizan ε = 0.01 según el protocolo documentado, por lo que conservan un 1 % de selección aleatoria de acciones y no son estrictamente greedy. 
Los valores se presentan como media ± desviación estándar entre episodios. Los hitos expresan la proporción de episodios que los alcanza al menos una vez, no la tasa de éxito de cada intento.

### Hardware y tiempo

- **CPU:** `AMD64 Family 23 Model 113`, PyTorch 2.14.1+cpu, Python 3.11.9.
- **Tiempo:** 2 706.68 s (≈ 45 min) para 2 000 000 de pasos, a **738.9 pasos/s** (`results/run_2m/timing.json`).
- **Por qué CPU y no GPU:** con observación RAM el cuello de botella es la CPU. El emulador ALE es
  secuencial, y la red es pequeña: copiar cada lote a la GPU cuesta casi lo mismo que el cómputo
  (`docs/notas-de-desarrollo.md`, sección 3). No medimos el tiempo en GPU.
- La estimación inicial fue de 1 a 3 horas. La medición del entrenamiento de 300 000 pasos fue de 6 min 21 s.

### Curva de aprendizaje

![Curva de aprendizaje](results/run_2m/learning_curve.png)

La curva muestra el retorno de cada episodio (raw, sin recorte), su promedio móvil de 50 episodios, la
evaluación greedy cada 25 000 pasos (media ± desviación estándar) y el máximo de filas por vida.

### Línea base a 300 000 pasos

## Interpretación de la curva de aprendisaje 

La curva reúne los retornos de 7 408 episodios de entrenamiento y 80 evaluaciones realizadas cada 25 000 pasos. Los retornos individuales muestran fluctuaciones, mientras el promedio móvil de 50 episodios permite observar la tendencia general. Este promedio suaviza la variación, pero no significa que todos los episodios tengan un desempeño similar.

Las evaluaciones muestran una mejora general: el retorno medio pasa de 6.0 puntos a los 25 000 pasos a 40.2 al finalizar los 2 millones. Sin embargo, el progreso no es monótono. El mayor retorno medio evaluado fue de 41.0 a los 1.7 millones de pasos; posteriormente disminuyó hasta 34.2 a los 1.925 millones y se recuperó al final. Por tanto, la curva respalda aprendizaje, pero no demuestra una convergencia definitiva ni una mejora continua.

En las últimas 20 evaluaciones, correspondientes al tramo posterior a 1.5 millones de pasos, las medias se encuentran entre 34.2 y 41.0 puntos. Esto indica que el agente mantiene un nivel de recompensa superior al observado al inicio, aunque todavía presenta variación entre evaluaciones.

También es necesario distinguir entrenamiento y evaluación. Según la configuración documentada, el entrenamiento mantiene ε = 0.05 después del periodo de decaimiento, mientras las evaluaciones periódicas utilizan ε = 0.01. En los últimos 50 episodios de entrenamiento, el retorno promedio fue de 34.32 puntos. Su diferencia con la evaluación final no implica una contradicción: ambos registros corresponden a condiciones de actuación distintas y no son directamente equivalentes.

Las barras de error representan la desviación estándar de los retornos de los 10 episodios de cada evaluación. Describen la dispersión entre partidas, no un intervalo de confianza ni la variabilidad entre entrenamientos independientes.

Evaluación greedy de 20 episodios (`results/baseline_300k/evaluation.json`):

| Política | Puntaje medio ± desv. estándar |
|---|---|
| Aleatoria | 9.45 ± 2.11 |
| DQN (300 000 pasos) | 25.20 ± 4.35 |

### Evaluación final frente a aleatoria

| Fuente | Episodios | Aleatoria | DQN a 2M pasos |
|---|---|---|---|
| Evaluación durante el entrenamiento (`results/run_2m/eval.csv`, última fila) | 10 | — | 40.2 ± 1.7 |
| Leaderboard por checkpoint (`results/run_2m/leaderboard.md`) | 20 | 9.4 ± 2.3 | 39.3 ± 2.9 |

Ambas fuentes usan semillas distintas, por eso los valores difieren ligeramente.

### Tabla de medallas

Cada medalla se marca si al menos el 50 % de los 20 episodios alcanzó el hito dentro de una vida
(`results/run_2m/leaderboard.md`):

| Checkpoint | Episodios | Puntaje | Filas máx. (media / máx.) | Casas (media / máx.) | 🐸 first_hop | 🥉 road_crossed | 🪵 first_log | 🥈 mid_river | 🥇 frog_home* | 🏆 three_homes* | 👑 level_cleared* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 20 | 9.4 ± 2.3 | 3.3 / 7 | 0.00 / 0 | 🐸 100% | 5% | 5% | 0% | 0% | 0% | 0% |
| 100,000 | 20 | 12.2 ± 1.9 | 3.5 / 7 | 0.00 / 0 | 🐸 100% | 5% | 5% | 0% | 0% | 0% | 0% |
| 250,000 | 20 | 20.6 ± 3.2 | 7.7 / 10 | 0.00 / 0 | 🐸 100% | 🥉 95% | 🪵 95% | 20% | 0% | 0% | 0% |
| 500,000 | 20 | 18.0 ± 2.0 | 8.5 / 9 | 0.00 / 0 | 🐸 100% | 🥉 90% | 🪵 90% | 🥈 90% | 0% | 0% | 0% |
| 1,000,000 | 20 | 32.0 ± 4.3 | 9.7 / 10 | 0.00 / 0 | 🐸 100% | 🥉 100% | 🪵 100% | 🥈 95% | 0% | 0% | 0% |
| 2,000,000 | 20 | 39.3 ± 2.9 | 9.9 / 10 | 0.00 / 0 | 🐸 100% | 🥉 100% | 🪵 100% | 🥈 100% | 0% | 0% | 0% |

Umbrales: 🐸 `first_hop` = fila 1; 🥉 `road_crossed` = fila 6; 🪵 `first_log` = fila 7; 🥈 `mid_river` = fila 9.
La fila de casa es la 12.

\* **Medallas proxy no verificadas.** La detección de casa no está validada contra trazas de RAM. Por eso
`frog_home`, `three_homes` y `level_cleared` son estimaciones y no logros confirmados del juego.

![Hitos por checkpoint](results/run_2m/milestones.png)

### Interpretación de los hitos de avance

La gráfica de hitos complementa el retorno acumulado al mostrar qué proporción de los 20 episodios evaluados alcanza determinadas etapas del juego. Estos porcentajes se calculan por episodio: alcanzar un hito al menos una vez no significa que el agente lo consiga en todas sus vidas o intentos.

Entre 100 000 y 250 000 pasos, la proporción de episodios que cruza la carretera aumenta del 5 % al 95 %. En ese mismo intervalo, alcanzar el primer tronco también pasa del 5 % al 95 %. Esto respalda una mejora importante en el avance hacia el río.

Entre 250 000 y 500 000 pasos, el hito de mitad del río aumenta del 20 % al 90 %, aunque el retorno medio disminuye de 20.6 a 18.0 puntos. Las métricas muestran aspectos distintos del comportamiento: obtener menos recompensa acumulada puede coincidir con alcanzar una etapa más avanzada. Los registros disponibles no permiten identificar una causa única de esta diferencia.

A los 2 millones de pasos, los 20 episodios evaluados alcanzan los hitos de carretera, primer tronco y mitad del río. Sin embargo, el máximo registrado continúa en 10 filas y no se reportan casas. Entre 1 y 2 millones de pasos, el retorno medio aumenta de 32.0 a 39.3, pero el máximo avance registrado no aumenta. Por eso distinguimos mejora de recompensa y ampliación del alcance del agente.

El hito first_hop aporta poca capacidad de comparación, porque también lo alcanza la política aleatoria en el 100 % de los episodios. Los indicadores frog_home, three_homes y level_cleared son proxies sin validación independiente y no deben interpretarse como logros confirmados del juego.


### Progresión en GIF

Primer episodio de evaluación de cada checkpoint (`results/run_2m/gifs/`):

<table>
<tr>
<td align="center"><b>Aleatoria</b><br><img src="results/run_2m/gifs/random.gif" width="200" alt="Aleatoria"></td>
<td align="center"><b>100 000</b><br><img src="results/run_2m/gifs/step_100000.gif" width="200" alt="100 000 pasos"></td>
<td align="center"><b>250 000</b><br><img src="results/run_2m/gifs/step_250000.gif" width="200" alt="250 000 pasos"></td>
</tr>
<tr>
<td align="center"><b>500 000</b><br><img src="results/run_2m/gifs/step_500000.gif" width="200" alt="500 000 pasos"></td>
<td align="center"><b>1 000 000</b><br><img src="results/run_2m/gifs/step_1000000.gif" width="200" alt="1 000 000 pasos"></td>
<td align="center"><b>2 000 000</b><br><img src="results/run_2m/gifs/step_2000000.gif" width="200" alt="2 000 000 pasos"></td>
</tr>
</table>

### Capturas del primer hito

| Hito | 2 000 000 pasos |
|---|---|
| Cruzar la carretera (`road_crossed`) | ![road_crossed](results/run_2m/screenshots/step_2000000_road_crossed.png) |
| Primer tronco (`first_log`) | ![first_log](results/run_2m/screenshots/step_2000000_first_log.png) |
| Mitad del río (`mid_river`) | ![mid_river](results/run_2m/screenshots/step_2000000_mid_river.png) |

### Observaciones

- **El aprendizaje ocurre por etapas.** Entre 100 000 y 250 000 pasos, cruzar la carretera pasa de 5 % a 95 %.
  Entre 250 000 y 500 000 pasos, llegar a la mitad del río pasa de 20 % a 90 %.
- **El puntaje oculta la habilidad.** A 500 000 pasos el puntaje medio bajó de 20.6 a 18.0, pero la medalla
  `mid_river` subió de 20 % a 90 %. El puntaje solo no muestra que el agente cruza el río con más frecuencia.
- **Techo en la fila 10.** El máximo de filas en una vida fue 10 a 1 000 000 y 2 000 000 de pasos. Ningún
  episodio llegó a la fila 12, así que no hay evidencia de casas (`homes_max = 0` en todos los checkpoints).
- **La medalla `first_hop` no discrimina.** La política aleatoria la alcanza en el 100 % de los episodios.

## Reflexión: qué fue lo más difícil

> Borrador dos de para la reflexión de grupo


Los resultados muestran que el agente aprendió a obtener mayor recompensa y a alcanzar etapas más avanzadas que la política aleatoria. 
En la evaluación por checkpoints, el retorno medio después de 2 millones de pasos fue de 39.3 puntos, frente a 9.4 de la política aleatoria. 
Esta mejora respalda el aprendizaje bajo el protocolo utilizado, aunque no significa que el agente haya resuelto Frogger.

El aprendizaje se manifestó por etapas. Primero aumentó la frecuencia con la que el agente cruzaba la carretera y después mejoró su avance hacia la mitad del río. 
Sin embargo, el retorno y los hitos no siempre evolucionaron en la misma dirección: a los 500 000 pasos, el puntaje medio disminuyó mientras aumentaba la proporción de episodios que alcanzaba la mitad del río.
Esta observación nos llevó a evaluar el progreso con ambas métricas, porque una caída del retorno no demuestra por sí sola una pérdida de aprendizaje.

También distinguimos obtener más recompensa de superar nuevas etapas. Entre 1 y 2 millones de pasos, el retorno medio aumentó, pero el máximo registrado permaneció en 10 filas y no se reportaron casas.
Por tanto, la evidencia muestra una mejora en el desempeño parcial, con una dificultad pendiente para completar el recorrido. 
Una posible explicación es que el agente acumule menos experiencia en las zonas finales, pero sería necesario analizar las visitas y pérdidas de vida por zona para comprobarlo.

Las evaluaciones periódicas presentan fluctuaciones incluso al final del entrenamiento. 
Aunque el desempeño se mantiene por encima del observado al inicio, no podemos afirmar una convergencia definitiva. 
Además, las desviaciones estándar describen diferencias entre episodios de evaluación; una sola corrida larga no permite establecer qué tan consistente sería el aprendizaje con otras semillas de entrenamiento.

Los hitos también tienen límites de interpretación. first_hop no distingue al agente entrenado de la política aleatoria, porque ambos lo alcanzan en todos los episodios evaluados. Los indicadores de casas son proxies cuya detección no fue validada contra trazas de RAM, por lo que no deben presentarse como logros confirmados.

Concluimos que el DQN mostró aprendizaje parcial y que combinar recompensa con hitos permitió comprender mejor su comportamiento.
Para fortalecer la evidencia, proponemos repetir entrenamientos con otras semillas, conservar la configuración y los resultados individuales de evaluación, y analizar las zonas donde se limita el avance antes de decidir qué cambios realizar en el agente.

<!-- BORRADOR: reescribir con la voz del grupo antes de entregar -->

> ✍️ Borrador para revisión del grupo.

Lo más difícil fue decidir qué observación usar con el plazo que teníamos. Los píxeles y una CNN eran la
versión clásica, pero requerían millones de cuadros que no cabían en el tiempo. Elegimos la RAM de 128 bytes
con una MLP. Ese cambio redujo el problema, aunque obligó a la red a descubrir por sí misma qué bytes
describen al sapo y a los autos.

La estimación de tiempo también fue un error. Pensábamos que el entrenamiento tomaría entre una y tres horas,
y el entrenamiento de 300 000 pasos en CPU terminó en 6 minutos. Aprendimos a medir antes de optimizar la
infraestructura.

Calibrar los hitos fue otra dificultad. Medimos las filas en cuadros renderizados y confirmamos el byte 90
de la RAM en cada paso de cinco episodios. La detección de casa nunca quedó verificada: el agente no pasó de
la fila 10, y la búsqueda en haz no encontró una trayectoria que llegara a la fila 12. Por eso las medallas de
casa son proxies, y así lo indicamos en las tablas.

La revisión independiente también nos enseñó algo. Su primera ejecución terminó con código de salida 0, pero
no leyó ningún archivo. Desde entonces verificamos el contenido de las salidas y no solo el código de salida.

Por último, el resultado de 500 000 pasos cambió nuestra lectura del progreso. El puntaje bajó, pero el
agente cruzaba el río con más frecuencia. Sin las medallas no habríamos visto esa mejora.

## Reproducción

Requisitos: [`uv`](https://docs.astral.sh/uv/). Las dependencias están fijadas en `uv.lock`.

```bash
uv sync --frozen                                          # instalar dependencias exactas
uv run pytest -q                                          # pruebas unitarias y de integración
uv run python train.py --smoke --output-dir runs/smoke    # entrenamiento de prueba (segundos)
uv run python train.py --total-steps 2000000 --output-dir results/run_2m   # corrida de 2M pasos
uv run python milestones.py --run-dir results/run_2m --episodes 20         # leaderboard y GIFs
uv run python milestones.py --run-dir results/run_2m --plot-only           # solo redibujar milestones.png
uv run python plot.py --input-dir results/run_2m          # curva de aprendizaje
uv run python evaluate.py --checkpoint results/run_2m/dqn_frogger.pt --episodes 20   # evaluación greedy
```

- `train.py` se niega a reutilizar un directorio que ya contiene una corrida. Para reemplazarla, usa
  `--overwrite`; solo borra los artefactos que el entrenador genera.
- Los checkpoints `*.pt` no se versionan (`.gitignore`). Se generan al entrenar y quedan en
  `results/<corrida>/checkpoints/` y en `dqn_frogger.pt`.
- `results/run_2m/` no incluye `config.json`, así que la configuración exacta de esa corrida se toma del
  comando anterior y de los valores por defecto de `config.py`.

## Estructura del proyecto

```
frogger-dqn-lab/
├── dqn_frogger/            # paquete del agente, sin dependencias de Stable-Baselines3
│   ├── agent.py            # red en línea y objetivo, ε-greedy y actualización de Bellman
│   ├── network.py          # MLP Q-network
│   ├── replay_buffer.py    # buffer circular con muestreo uniforme
│   ├── schedule.py         # decaimiento lineal de ε
│   ├── config.py           # hiperparámetros
│   ├── trainer.py          # bucle de entrenamiento, registros y checkpoints
│   ├── env.py              # fábrica del entorno con RAM normalizada
│   ├── milestones.py       # rastreador de filas, casas y medallas
│   ├── leaderboard.py      # estadísticas y tablas de medallas
│   └── evaluation.py       # episodios de evaluación
├── train.py                # entrada de entrenamiento
├── evaluate.py             # evaluación de un checkpoint
├── milestones.py           # leaderboard por checkpoint (GIFs y capturas)
├── plot.py                 # curva de aprendizaje
├── tests/                  # pruebas con pytest
├── results/
│   ├── run_2m/             # corrida de 2M pasos: leaderboard, curvas, GIFs y capturas
│   ├── baseline_300k/      # línea base de 300 000 pasos
│   └── calibration/        # calibración de filas con cuadros renderizados
├── docs/
│   ├── notas-de-desarrollo.md   # decisiones, medición, calibración y revisión independiente
│   └── codex-review.md          # informe de la revisión independiente
└── odd/tasks/dqn-frogger.md     # seguimiento de tareas
```

## Documentación

- [Notas de desarrollo](docs/notas-de-desarrollo.md): decisiones (RAM frente a píxeles, CPU frente a GPU),
  calibración de hitos y revisión independiente.
- [Informe de la revisión independiente](docs/codex-review.md): hallazgos y resoluciones.
