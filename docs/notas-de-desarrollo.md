# Notas de desarrollo

Registro de decisiones y problemas encontrados durante el Taller 2 (DQN). Sirve como insumo
para la sección de reflexión del README.

## 1. Un repositorio por entrega

El taller anterior (PPO sobre `ALE/Frogger-v5`) usa Stable-Baselines3 y vive en su propio
repositorio. Este taller exige un DQN escrito por el grupo, así que se creó un repositorio
independiente para que cada enlace entregado contenga un solo trabajo con su propio README.

## 2. Observación RAM en lugar de píxeles

El DQN clásico (Mnih et al., 2015) usa frames de 84×84 apilados y una CNN, y necesita
millones de pasos. Con el plazo disponible se eligió `obs_type="ram"`: un vector de 128 bytes
con el estado interno de la consola, procesado por una MLP. El juego es el mismo; cambia la
representación del estado.

Costo de la decisión: la RAM no tiene estructura espacial explícita y algunos bytes son
irrelevantes, así que la red debe descubrir qué posiciones codifican al sapo y a los autos.

## 3. ¿Dónde entrenar? CPU vs GPU vs Colab

Hardware disponible: PC con GTX 1660 SUPER (6 GB), Mac M4 y Colab (posible A100).

Hallazgo: con observación RAM el cuello de botella es la **CPU**, no la GPU.

- Cada paso del agente ejecuta el emulador de Atari (ALE), que es secuencial y corre en un núcleo.
- La red es pequeña (128 → 256 → 256 → acciones) y se actualiza con lotes de 64; copiar el lote
  a la GPU cuesta casi lo mismo que el cómputo.
- Por eso una A100 en Colab no acelera este caso: los vCPU de Colab son más lentos que un núcleo
  de un M4. La GPU solo sería decisiva en la variante con píxeles y CNN.

Plan: entrenar en la PC con CPU, medir el rendimiento a los ~10 minutos y, si es demasiado
lento, mover el entrenamiento al Mac M4.

### Medición en la PC

La estimación inicial (1–3 horas) fue muy pesimista. El entrenamiento de 300 000 pasos en
CPU terminó en **6 min 21 s (~790 pasos/s)**, sin necesidad de GPU, Colab ni el Mac. La
lección: medir antes de optimizar la infraestructura.

| Política (20 episodios) | Puntaje medio | Desv. estándar |
|---|---|---|
| Aleatoria | 9.4 | 2.1 |
| DQN (300k pasos) | 25.2 | 4.4 |

La evaluación greedy siguió subiendo hasta el último punto (6.0 → 25.0), así que el agente no
había convergido: hay margen para un entrenamiento más largo.

## 4. Calibración de hitos

Objetivo: fijar con evidencia cuántas filas de avance corresponden a cada hito (cruzar la
carretera, entrar al río, llegar a casa). Método y archivos en `results/calibration/`
(`calibrate.py`, `row_*.png`, `rows_montage.png`). Se usó el checkpoint de 300k pasos y
episodios greedy con ε = 0.01.

### Mapa de filas (medido en cuadros renderizados)

El sapo se localizó por color (`RGB 110,156,66`) en cada cuadro. Su centro vertical cae en
bandas de ~13 px. Fila 0 es la franja de inicio; la numeración sigue hacia arriba.

| Filas | Zona | Evidencia |
|---|---|---|
| 0 | Franja de inicio (amarilla, abajo) | Centroide y≈174 al inicio |
| 1–5 | Carretera (5 carriles) | Columna x=20: bandas negras entre y≈103 y y≈169 |
| 6 | Franja mediana (segura) | Cuadro del paso 119 (`row_06.png`) |
| 7–11 | Río (5 filas; troncos en 7–11) | Primera fila de río en el paso 121 (`row_07.png`) |
| 12 | Fila de casas (5 huecos entre arbustos) | Columna en x=48 y x=80, y≈11–24 |

Medallas por umbral de fila (una vida): `first_hop` = fila 1; `road_crossed` = fila 6;
`first_log` = fila 7; `mid_river` = fila 9 (centro del río).

### Lectura de RAM (verificada)

- **Byte 90** = filas alcanzadas en la vida actual. Se reinicia al perder una vida.
  Coincide en todos los pasos de los 5 episodios (257/257, 257/257, 257/257, 238/238,
  259/259).
- **Byte 78** = puntaje acumulado en BCD (0x09 + 1 = 0x10). No se reinicia al morir.
  Coincide en todos los pasos (mismo conteo).

El conteo por recompensa (+1 por fila nueva) reproduce el byte 90. Por eso el rastreador usa
solo recompensa y pérdida de vida; no necesita RAM.

### Alcance del agente de 300k pasos

En 5 episodios greedy (semilla 1000–1004) el agente alcanzó como máximo la fila 8 en una
vida (6, 8, 3, 2, 2 filas por vida en el primer episodio). Muere en el río o al cruzar el
carril siguiente. Solo algunas vidas llegan a la franja mediana (fila 6, `road_crossed`); las
demás mueren antes. Ninguna muestra alcanzó la fila 9 ni la 12.

### Detección de "sapo en casa": no confiable

Se intentó encontrar una secuencia de acciones que llegue a la fila 12 con una búsqueda en
haz sobre estados clonados del emulador (anchura 200, 450 niveles de profundidad, sin perder
vidas). La mejor trayectoria llegó a la fila 11 y no llegó a casa. Por el límite de tiempo
asignado no se amplió la búsqueda.

Consecuencia: no hay evidencia de que llegar a casa dé recompensa ni de que el sapo vuelva al
inicio sin perder una vida. El rastreador usa el respaldo indicado en el plan: **llegar a la
fila 12 cuenta como una casa** (y reinicia la fila a 0 sin perder vida). Esa regla es una
suposición sin verificar. Las medallas `frog_home`, `three_homes` y `level_cleared` quedan
marcadas como no verificadas hasta que una trayectoria real llegue a casa.

## 5. Revisión independiente

**Primera ejecución fallida.** La primera revisión con Codex falló por el entorno de Windows: el
sandbox de solo lectura devolvió `helper_unknown_error`. El proceso terminó con código de salida
0, pero no leyó ningún archivo. Lección: verificar el contenido de la salida de la herramienta,
no solo el código de salida.

**Segunda ejecución.** La revisión recibió el código fuente por entrada estándar. El informe
completo está en `docs/codex-review.md`.

**Hallazgos y resoluciones** (detalle en la sección «Resolution» de ese documento):

1. Reutilizar un directorio de salida mezclaba checkpoints de distintas ejecuciones. Ahora el
   entrenamiento se niega a usar un directorio con una ejecución previa; `--overwrite` borra
   solo los artefactos conocidos. El leaderboard toma los checkpoints del `config.json`.
2. Las medallas de casa son un proxy no verificado. Una casa ya no cuenta si la rana muere en el
   mismo paso. Las medallas `frog_home`, `three_homes` y `level_cleared` se marcan como proxy en
   `leaderboard.json` y `leaderboard.md`. Falta validar la detección contra trazas de RAM.
3. Faltaban datos para reproducir una ejecución. `config.json` se escribe al iniciar, y las
   semillas de entrenamiento dentro del rango de evaluación del leaderboard se rechazan.
4. Faltaban pruebas del entrenador y de los medios. Se añadieron pruebas con un entorno
   simulado (terminación frente a truncamiento, sincronización del objetivo, recompensas
   recortadas) y pruebas de lectura de PNG y GIF.
5. La política de evaluación dependía del historial. Ahora cada episodio usa su propio generador
   aleatorio, derivado de la semilla del episodio.
6. `leaderboard.json` omitía `homes_max`. Ahora se incluye y el esquema tiene una prueba.

**Corrección sobre dependencias.** La revisión afirma que las dependencias no están congeladas.
Eso es solo parcialmente cierto: `uv.lock` ya estaba versionado y fija versiones exactas (por
ejemplo `gymnasium` 1.4.0, `ale-py` 0.12.1 y `torch` 2.14.1+cpu). `pyproject.toml` declara
rangos, pero el lockfile controla las versiones instaladas. No se cambiaron las restricciones.

## 6. CPU frente a GPU: la medición

La sección 3 concluyó, sin medir, que la GPU no ayudaría. Al medirlo, el resultado fue más fuerte:
la GPU fue **más lenta**.

Método: un entorno aparte con PyTorch 2.11.0+cu128 (para no tocar el proyecto, que usa la versión
solo CPU), misma configuración y misma semilla en ambos dispositivos. Script:
`benchmarks/cpu_vs_gpu.py`; resultados: `results/benchmark_cpu_gpu.json`.

| Medición | CPU | GTX 1660 SUPER |
|---|---|---|
| Emulador solo, sin red | 2 191.4 pasos/s | — |
| Una actualización (lote de 64) | 2.135 ms | 3.371 ms |
| Entrenamiento de 100 000 pasos | 133.0 s (751.8 pasos/s) | 172.3 s (580.6 pasos/s) |
| Evaluación a 100 000 pasos | 8.3 ± 2.3 | 11.9 ± 1.8 |

Lecciones:

- Con una red de ~100 000 parámetros y lotes de 64, el costo fijo de mover datos a la GPU y lanzar
  kernels supera al cálculo. "Tener GPU" no implica "entrenar más rápido".
- La diferencia de puntaje es ruido de una corrida corta con una semilla: CPU y GPU no producen
  exactamente los mismos números en coma flotante, así que las trayectorias se separan.
- Medir antes de decidir la infraestructura, otra vez: la conclusión razonada era correcta en la
  dirección, pero el dato real (23 % más lenta) es más informativo que la suposición.

## 7. Reto: píxeles y diseño mixto CPU/GPU

Objetivo personal, fuera de la entrega: superar el techo de la fila 10 del DQN con RAM y MLP.
La hipótesis es que la red necesita la imagen (posiciones y velocidades) y no solo 128 bytes de
RAM. La variante con píxeles usa CPU para emular y preprocesar, y GPU para la CNN y el aprendizaje.

### Hallazgos de `ale_py.vector_env.AtariVectorEnv` (ale-py 0.12.1)

Medidos con sondas en la máquina de trabajo; el código está en `dqn_frogger/pixels/vector_env.py`.

| Aspecto | Observado |
|---|---|
| Juego | `game="frogger"` (id de gym `ALE/Frogger-v5`) |
| Observación | `uint8` de forma `(N, 4, 84, 84)`: 4 cuadros en gris, 84×84 |
| Acciones | `Discrete(5)`; el entorno RAM de gym también reporta 5 |
| Vidas | `info["lives"]` empieza en 4 y llega a 0 en el paso terminal |
| Terminación | `terminated=True` con vidas en 0; es la observación terminal real |
| Truncamiento | `truncated=True` al superar `max_num_frames_per_episode` (108 000 por defecto) |
| Autoreset `NextStep` | El paso después de un fin devuelve la observación de reinicio, recompensa 0, sin fin. La acción se ignora |
| Autoreset `SameStep` | La observación de reinicio llega en el mismo paso terminal; la terminal solo aparece en `info["final_obs"]` (visto en truncamiento) |
| `Disabled` | No aceptado por ale-py 0.12.1 |
| Render RGB | No existe en el entorno vectorial |

Consecuencias para el diseño:

- Se usa `NextStep`, que conserva la observación terminal exacta. El paso de reinicio cuenta
  para el presupuesto de pasos, pero no se guarda ni se muestra al rastreador de hitos.
- Las medallas y las GIFs no pueden salir del entorno vectorial. La reproducción usa un entorno
  de gym con RGB y `AtariPreprocessing`. No es bit a bit igual al pipeline de entrenamiento:
  el reinicio con no-op y fuego, y el RNG de acciones pegajosas, difieren. Dos semillas de
  prueba divergieron en recompensas y terminales dentro de 400 pasos, así que la reproducción
  no puede ir paso a paso junto al entorno vectorial.
- La semilla de `reset` debe ser un entero o un arreglo con una semilla por entorno.

### Memoria del búfer

Cada transición guarda `obs` y `next_obs` como `uint8` (4, 84, 84): 2 × 28 224 bytes. Con una
capacidad de 200 000 transiciones son unos **11,3 GB** de RAM, el doble de lo que sugiere
solo un arreglo de observaciones. Cabe en 32 GB, pero el dato correcto es 11,3 GB.

### Verificación de la infraestructura

- `torch` 2.11.0+cu128 con `torch.cuda.is_available() == True`; dispositivo
  NVIDIA GeForce GTX 1660 SUPER.
- Pruebas: 70 pasan (54 de la RAM más 16 nuevas: red, Double DQN, n-step, búfer, entrenador).
- Humo de 2 000 pasos (4 entornos, 1 evaluación de 1 episodio). Orientativo, no es una medición:

| Dispositivo de la CNN | Pasos/s | Entorno | Acción | Aprendizaje | Evaluación |
|---|---|---|---|---|---|
| GPU (GTX 1660 SUPER) | 202.3 | 0.56 s | 0.79 s | 3.22 s | 5.21 s |
| CPU | 104.1 | 0.50 s | 0.88 s | 12.37 s | 5.35 s |

En el humo el aprendizaje en GPU ya es casi cuatro veces más rápido que en CPU (≈7 ms y ≈27 ms
por actualización de lote 32). La evaluación domina el tiempo total porque corre con un solo
entorno y un episodio largo; el pilotaje de la sección H2 medirá el régimen real.
