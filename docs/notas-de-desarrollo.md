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
