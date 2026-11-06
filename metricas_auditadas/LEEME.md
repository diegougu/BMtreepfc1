REVISION v03: lee primero AUDITORIA.md. Se corrige asignacion de centros semienteros; el aprendizaje sigue siendo experimental.

# Ocho métricas comunes: BMTree y FlexFlood

Este paquete ejecuta ambos índices secuencialmente sobre los mismos puntos,
consultas de entrenamiento/prueba y operaciones. Imprime una tabla final y
conserva resultados en JSON, CSV y texto. Es una demo 2D en RAM, sin PostgreSQL.
No reemplaza `adaptive_bmtree` ni los archivos del repositorio original de FlexFlood.

## Instalar y ejecutar en WSL

Copia `metricas_auditadas` junto a `adaptive_bmtree`, dentro de:

```
C:\Users\HP\Documents\Learned-BMTree-main\Learned-BMTree-main
```

```bash
cd /mnt/c/Users/HP/Documents/Learned-BMTree-main/Learned-BMTree-main
source ~/venvs/bmtree/bin/activate
python -m pip install -r metricas_auditadas/requirements.txt
python metricas_auditadas/run.py --out comparacion_metricas_01
```

Si falta el compilador: `sudo apt install -y g++`.
El entrenamiento de FlexFlood ahora necesita `scikit-learn`; su instalación se
incluye en requirements. No requiere GPU.

La salida termina con `METRICAS FINALES` y dos columnas, BMTree y FlexFlood.
El programa muestra cinco mensajes de progreso antes de la tabla.

```bash
cat comparacion_metricas_01/metricas.txt
```

- `metricas.txt`: resumen listo para leer.
- `metricas.csv`: las ocho métricas y sus unidades, para Excel.
- `resultados.json`: configuración, métricas crudas, fases, checksums, hashes de
  entrada, versiones y detalles del aprendizaje/adaptación.
- `bmtree_raw.json` y `flexflood_raw.json`: mediciones directas de C++.
- `*_stderr.txt`: mensajes de error de cada ejecutable.

Para repetir, usa otro `--out`; se rechaza sobrescribir resultados anteriores.
La ejecución normal ya valida contra fuerza bruta. Para las pruebas adicionales:

```bash
python -m unittest discover -s metricas_auditadas -p 'test_*.py' -v
```

## Qué mide cada fila

| Métrica | Definición |
|---|---|
| Latencia media de consulta | Tiempo de todas las búsquedas medidas / cantidad de búsquedas; microsegundos. Incluye localizar candidatos, filtrar y materializar las coordenadas de resultados. Se mide tras las actualizaciones y el mantenimiento. |
| QPS | Búsquedas completadas / segundos del mismo tramo. En este benchmark secuencial es el recíproco de la latencia media. |
| Tiempo medio amortizado de actualización | Tiempo de eliminaciones e inserciones, incluido mantenimiento, / número de operaciones. En BMTree suma controlador Python y recodificación; en FlexFlood el mantenimiento ya está dentro de los updates y NO se vuelve a sumar. |
| Tiempo de adaptación | BMTree: tiempo del controlador (detección + selección + MCTS) más recodificación C++. FlexFlood: tiempo acumulado actualizando contadores, comprobando condiciones y ejecutando split/merge/equalize. Es una medición por componentes, no tiempo hasta lograr una calidad objetivo ni latencia de un servicio online. |
| Degradación de latencia | 100 × (latencia final / latencia inicial − 1). Misma lista de consultas antes y después, misma cardinalidad; negativo es mejora. La selectividad real puede cambiar con los datos. |
| Recall (micro) | Suma de resultados correctos recuperados / suma de resultados esperados. Se contrasta con fuerza bruta después de actualizar. Si todas las respuestas esperadas están vacías se define como 1. También se exigen cero falsos positivos e igualdad de los resultados, incluidos casos vacíos, antes y después. |
| Tiempo total de construcción | BMTree: aprendizaje inicial medido en Python + construcción del índice C++. FlexFlood: constructor completo, con muestras de costo, entrenamiento de sus Random Forests, elección de particiones y carga de puntos. Excluye generación/lectura/exportación de archivos, compilación y validación. |
| Memoria del índice residente | Estimación estructural en MiB al final. Suma objetos, capacidades de vectores y bytes de nodos B-tree. Incluye los registros almacenados por cada índice y metadatos. Excluye buffers comunes de entrada, resultados temporales, memoria temporal de entrenamiento y sobrecarga interna del asignador. NO es RSS ni pico de RAM del proceso. |

La estimación de memoria compara estructuras completas residentes con su propio
almacenamiento de puntos. No debe describirse en el paper como memoria adicional
excluyendo datos base. En BMTree incluye las posiciones reservadas y los registros
eliminados que esta implementación conserva para mantener IDs estables.

Todos los algoritmos devuelven las mismas coordenadas en un vector nuevo por
consulta. Esto evita comparar la materialización de IDs en un índice con la
materialización de coordenadas en el otro.

La validación exhaustiva, el cómputo de memoria y los archivos quedan fuera de
los intervalos de consulta. Ambos reciben calentamiento de caché. Los relojes
son monotónicos. El instrumentation del mantenimiento introduce una pequeña
sobrecarga, incluida en las actualizaciones de FlexFlood.

## Dataset de prueba y presupuestos

- 10.000 puntos enteros únicos, representados exactamente como doubles en ambos
  motores C++; dos dimensiones de 20 bits.
- Cambio localizado de distribución, cardinalidad constante: se borran los
  puntos elegidos y luego se insertan sus reemplazos, en el mismo orden para ambos.
- No hay duplicados de coordenadas: FlexFlood original tiene semántica de conjunto.
  Este paquete rechaza duplicados en vez de comparar cardinalidades diferentes.
- 240 consultas para aprendizaje y 60 distintas para prueba. Se generan una sola
  vez para ambos; la prueba no se utiliza para entrenar.
- 1.000 repeticiones de esas 60 consultas: no son 60.000 consultas diferentes.
- BMTree aprende desde Z-order; muestra de hasta 2.000 puntos, cuatro niveles y
  32 rollouts. Reutiliza la reconstrucción independiente v0.1, con las mismas
  limitaciones de fidelidad ya documentadas en el paquete anterior.
- FlexFlood usa el aprendizaje original basado en tres RandomForestRegressor.
  Para que la demo sea rápida: 20 configuraciones muestreadas y dos segundos de
  búsqueda de configuración (el tiempo total de construcción puede ser mayor).
  Este presupuesto no pretende reproducir el del paper.
- Se fijan semillas de datos y búsqueda; la selección de FlexFlood utiliza costos
  medidos y un límite temporal, por lo que su configuración y resultados pueden
  variar entre ejecuciones aun con la misma semilla.
- Los parámetros no se ajustan buscando favorecer un índice en los resultados de prueba.

Ejemplo de otra corrida, misma escala y semilla diferente:

```bash
python metricas_auditadas/run.py --seed 43 --out comparacion_metricas_02
```

Para avanzar a 100.000 puntos, una vez comprobada la demo:

```bash
python metricas_auditadas/run.py --points 100000 --out comparacion_100k_01
```

El orden de ejecución de esta demo es BMTree y después FlexFlood. Para un estudio
final deben alternarse los órdenes y realizarse corridas independientes; estas
cifras no sustituyen el protocolo final ni permiten extrapolar a OSM.

## Procedencia y cambios de instrumentación

FlexFlood proviene del ZIP suministrado por el usuario, commit de archivo
`5964041505286adb5c2bd2307e7c933771af4d29`. Se incluye su licencia en `flex/LICENSE`
y las licencias originales de cpp-btree. No se usan PostgreSQL ni APIs externas.

Cambios en la copia de FlexFlood:

1. Contador de tiempo del mantenimiento y número de split/merge/equalize.
2. Método de estimación de bytes residentes.
3. Corrección del orden de argumentos entre el constructor y `learn` (tiempo y
   número de muestras estaban intercambiados).
4. Esperar la terminación del proceso Python y comprobar su código de salida
   antes de abrir/leer la configuración aprendida. El código original leía la
   configuración antes de cerrar el proceso y podía leer un archivo anterior.
5. Comillas en las rutas para admitir directorios con espacios, semilla de
   búsqueda Python y temperatura mínima positiva para costos medidos nulos.

No se cambiaron los algoritmos ni umbrales de split/merge/equalize de esa versión.
Las diferencias que pueda tener el código distribuido frente al paper permanecen;
esto no constituye una reproducción exacta del artículo.

La copia de entrenamiento usa `flex/train/sample.csv` y `train.csv`. El ejecutor
usa un bloqueo para evitar dos ejecuciones simultáneas en el mismo paquete.
Estos archivos se generan al ejecutar; no necesitan descargarse por separado.

## Archivos para localizar la medición

- `common.hpp`: consultas cronometradas, comparación con fuerza bruta y salida JSON.
- `bmtree_metrics.cpp`: construcción, actualización, recodificación y memoria de BMTree.
- `flexflood_metrics.cpp`: las fases equivalentes de FlexFlood.
- `flex/src/FlexFlood.hpp`: medición interna del mantenimiento y memoria.
- `run.py`: dataset común, aprendizaje BMTree, ejecución, unidades y tabla final.

Si una consulta no coincide con fuerza bruta, se conservan los resultados crudos
para diagnóstico y el programa termina con error antes de publicar una tabla de
comparación aparentemente válida.
