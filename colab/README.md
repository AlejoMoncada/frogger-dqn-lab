# Notebook de Colab: DQN de píxeles (overkill)

`frogger_overkill.ipynb` entrena el DQN de píxeles sobre Frogger en Google Colab y guarda todo en Google Drive.

## Abrir el notebook

Abra esta URL en el navegador:

https://colab.research.google.com/github/AlejoMoncada/frogger-dqn-lab/blob/exp/home-challenge/colab/frogger_overkill.ipynb

## Ejecutar

1. Seleccione un runtime con GPU A100 en *Entorno de ejecución > Cambiar tipo de entorno de ejecución*. No está garantizado: revise la celda de hardware.
2. Opcional: edite `USER_OVERRIDES` en la celda de configuración.
3. *Entorno de ejecución > Ejecutar todo*. Autorice el acceso a Google Drive cuando se le pida.

## Reanudar tras una desconexión

Vuelva a ejecutar todas las celdas. Si existe `checkpoints/latest.pt` en la carpeta de Drive, el entrenamiento reanuda desde ahí. Los checkpoints se guardan cada 250.000 pasos, así que una desconexión pierde como máximo 250.000 pasos. El buffer de replay no se guarda y se rellena de nuevo.

## Coste

Las unidades de cómputo de Colab se consumen mientras la sesión está activa, y un entrenamiento de 20 M de pasos puede agotar una cuota completa. Compruebe su saldo antes de empezar.
