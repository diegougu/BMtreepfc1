# BMTree v04 + FlexFlood: comparación 2D en RAM

Esta carpeta reemplaza a metricas_auditadas para nuevas ejecuciones. Conserva las
anteriores como respaldo; NO mezcles archivos entre versiones. Las ocho métricas
siguen iguales. Lee FIDELIDAD.md para los cambios, correspondencia con el paper
y decisiones explícitas. El código de FlexFlood se conserva de v03.

Extrae metricas_bmtree_v04 dentro de tu carpeta Learned-BMTree-main:

```bash
cd /mnt/c/Users/HP/Documents/Learned-BMTree-main/Learned-BMTree-main
source ~/venvs/bmtree/bin/activate
python -m pip install -r metricas_bmtree_v04/requirements.txt
python -m unittest discover -s metricas_bmtree_v04 -p 'test_*.py' -v
python metricas_bmtree_v04/run.py --out comparacion_v04_01
```

No necesitas instalar PostgreSQL ni PyTorch. Requiere Python, numpy, scikit-learn
y g++ con C++17. El ejecutor compila ambos índices automáticamente.
Si el directorio de resultados existe con contenido, usa otro --out.

## Qué esperar

10.000 puntos 2D sintéticos comunes, inicialmente uniformes; cambio localizado,
240 consultas de entrenamiento y 60 de prueba separadas. El cronómetro repite
las mismas 60 consultas 1000 veces. Se verifican respuestas antes y después.
Recall micro usa suma de aciertos / suma de esperados, y además se exige igualdad
exacta y cero falsos positivos. Si todas las consultas están vacías se define
recall=1 por convenio, sin saltarse la igualdad exacta.

El programa indica si se ejecutó reentrenamiento. NO es un fallo que indique NO:
puede no haber regiones que cumplan el filtro y el presupuesto. En ese caso el
tiempo de adaptación corresponde a detección/control y llamada de mantenimiento,
no a entrenamiento. No compares ese tiempo con otra corrida que sí reentrene
como si ambas hubieran hecho el mismo trabajo.

Se guardan metricas.txt, metricas.csv y resultados.json, además de árboles,
datos de entrada, métricas crudas y errores de ejecutables. El JSON registra
GAS, máximo, estados explorados, parámetros y selección de regiones.

## Parámetros explícitos

--points 10000, --sample 2000, --levels 4, --rollouts 32, --seed 42,
--shift-threshold 0.1, --retrain-budget 0.5, --detect-depth 5, --block-size 50,
--repeats 1000, --flex-learning-seconds 2, --flex-samples 20.

levels es el número de transiciones aprendidas; los bits restantes se completan
con Z-order. Más niveles/simulaciones implican más coste y no garantizan menores
tiempos de consulta. No son los presupuestos de los experimentos originales.

## Auditorías adicionales

```bash
python metricas_bmtree_v04/audit_independent.py
python metricas_bmtree_v04/audit_original_encoder.py bmtree/tree.py
```

La segunda orden requiere el repositorio original que ya descargaste.

## Interpretación de métricas

Latencia y QPS: materialización y filtrado incluidos, un hilo, datos después de
actualizar. QPS es recíproco de latencia en este benchmark secuencial.
Actualización amortizada: inserciones + eliminaciones + mantenimiento dividido
por operaciones. En FlexFlood el mantenimiento ya está dentro de updates; no se
suma dos veces. En BMTree incluye controlador y recodificación.
Construcción: entrenamiento + índice, sin compilación ni generación de datos.
Memoria: estructuras residentes propias incluyendo registros; no RSS, no pico.
Degradación: 100*(latencia_final/latencia_inicial - 1), combina cambio de datos,
selectividad y adaptación; no aísla beneficio del reentrenamiento.

Estos son resultados de una reimplementación independiente en RAM, no una
reproducción idéntica de todos los experimentos de los artículos.
