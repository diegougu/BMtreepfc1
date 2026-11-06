# BMTree adaptativo en memoria — reconstrucción independiente v0.1

Este directorio amplía la funcionalidad disponible en Learned-BMTree. No modifica
los archivos originales, no usa PostgreSQL y NO es código oficial de los autores
ni una reproducción experimental ya validada del paper ampliado.

Referencia implementada: *BMTree: Designing, Learning, and Updating Piecewise
Space-Filling Curves for Multi-Dimensional Data Indexing*, especialmente secciones
V y VI, ecuaciones 2–6 y algoritmos 1–2, PDF proporcionado por el usuario.
Repositorio base inspeccionado: https://github.com/gravesprite/Learned-BMTree,
commit `230ead0d82489b2f199baf165b0efa5f0c99b8c7`.

## Ejecutar en tu WSL

Copia **solo la carpeta `adaptive_bmtree`** del ZIP dentro del repositorio:

```
C:\Users\HP\Documents\Learned-BMTree-main\Learned-BMTree-main\adaptive_bmtree
```

En WSL:

```bash
cd /mnt/c/Users/HP/Documents/Learned-BMTree-main/Learned-BMTree-main
source ~/venvs/bmtree/bin/activate
sudo apt update
sudo apt install -y g++
python -m pip install numpy
python -m unittest discover -s adaptive_bmtree -p 'test_*.py'
python adaptive_bmtree/run.py --demo --out prueba_adaptativa_01
```

El último comando genera una prueba sintética reproducible de 10 000 puntos,
modifica parte de los datos manteniendo su cardinalidad, detecta cambios,
reentrena regiones y mide el índice en C++. No necesitas instalar una BD ni CUDA.
Si vuelves a ejecutar, usa otro `--out` para conservar las mediciones anteriores.

Salida principal: `prueba_adaptativa_01/results.json`.
`old_tree.txt` y `new_tree.txt` permiten inspeccionar el árbol antes/después.
`split.json` deja registro de la separación entrenamiento/prueba.

La prueba de equivalencia con el repo original necesita sus dependencias ya
instaladas, incluyendo PyTorch; si faltan se marca explícitamente como omitida.
Las otras pruebas y el ejecutor nuevo solo requieren NumPy y un compilador C++17.

## Qué está implementado

| Parte | Archivo / función | Comportamiento |
|---|---|---|
| Leer el árbol del repo | `adaptive.py: read_tree` | Formato de siete columnas; admite hijos divididos y compartidos. |
| Codificación SFC | `encode`, `Tree::key` | Consume bits de mayor a menor; termina con Z-order donde no hay acción aprendida. |
| Cambios en datos | `detect`, `js` | Histograma sobre dos niveles, divergencia Jensen–Shannon. |
| Cambios en consultas | `query_clusters`, `detect` | Centros de consultas, agrupación por log-área y log-relación de aspecto. |
| Prioridad de regiones | `detect`, `select` | Diferencia de ScanRange medio, BFS, OP descendente por nivel y presupuesto espacial estricto. |
| Aprendizaje parcial | `train_forest` | Acciones conjuntas dimensión/división, UCT, simulación por terminación Z y recompensa ScanRange. |
| Ampliación del reentrenamiento | `adapt` | Añade regiones no tocadas si la mejora de entrenamiento es menor que 1%; máximo ocho rondas y mismo presupuesto acumulado. |
| Índice en RAM | `benchmark.cpp: Index` | Claves `(SFC, id)` en cpp-btree, inserción, eliminación por ID y consulta exacta. |
| Recodificación parcial | `Index::rekey` | Busca los intervalos de las regiones elegidas, retira las claves antiguas e inserta las nuevas. |
| Validación | `Index::verify`, `test_adaptive.py` | Compara IDs completos contra fuerza bruta, conserva duplicados y detecta claves desactualizadas. |

El B-tree es el contenedor ordenado de las claves que produce BMTree. Utilizar
este contenedor en RAM no implica una integración con una base de datos.
Los headers de cpp-btree se toman del ZIP de FlexFlood proporcionado por el
usuario y conservan su licencia Apache-2.0 en `vendor/COPYING`.

## Optimizaciones implementadas

- Codificación vectorizada con NumPy durante el entrenamiento.
- Reutilización de claves de puntos fuera de las regiones seleccionadas.
- Caché de evaluaciones repetidas de un mismo árbol.
- Búsqueda binaria sobre límites de bloques para el objetivo ScanRange.
- Compilación C++17 `-O3 -DNDEBUG`; acceso a nodos mediante vector.
- Búsquedas por intervalo SFC, filtrado geométrico y buffers de resultados reutilizados.
- Recodificación limitada a las regiones afectadas; extracción de IDs antes de
  mutar el B-tree para evitar invalidar iteradores.
- Muestreo reproducible y límites explícitos del costo de búsqueda MCTS.
- Se conserva el mejor árbol observado según entrenamiento: un candidato peor
  en ese objetivo no sustituye al árbol previo. Esto NO garantiza mejor latencia
  real ni mejor rendimiento en consultas no vistas.

## Métricas y límites de interpretación

`results.json` incluye:

- `query_before_us`, `query_shifted_us`, `query_adapted_us`: tiempo amortizado
  por consulta antes del cambio, tras cambiar datos y tras adaptar el árbol.
- `qps_before`, `qps_shifted`, `qps_adapted`: consultas/segundo de esas fases.
  En esta ejecución secuencial QPS y latencia media son recíprocos.
- `update_without_adaptation_us`: tiempo de inserciones y eliminaciones / operaciones.
- `update_with_adaptation_us`: (actualizaciones + adaptación) / operaciones.
- `adaptation_seconds`: tiempo del controlador Python, incluyendo detección y
  entrenamiento, más tiempo C++ de recodificación. Es una **suma de componentes
  de un experimento por lotes**, no una medición continua de un servicio online.
- `degradation_shifted_pct`, `degradation_adapted_pct`: 100 × (latencia de la
  fase / latencia inicial − 1). Un valor negativo representa mejora.
- `exact_results_verified`: compara resultados completos contra fuerza bruta;
  comprueba tanto ausencias como falsos positivos, incluidas coordenadas duplicadas.

La lectura/exportación de archivos, compilación y validación exhaustiva no se
incluyen en los tiempos del índice. El tiempo de construcción sí se reporta
aparte; el aprendizaje inicial, cuando se solicita, aparece en `initial_training`.
La carga de consulta medida devuelve IDs materializados en un vector. No se
compara ese tiempo con una implementación que solamente cuente resultados.
Se calienta la caché en cada fase. Reporta este protocolo en el paper.

El benchmark conserva cardinalidad y ejecuta exactamente las mismas consultas
retenidas en las tres fases. El número de resultados puede cambiar con la nueva
distribución: por ello la degradación también puede reflejar cambios de
selectividad. Para aislar el beneficio de adaptación compara especialmente
`shifted` frente a `adapted`: misma distribución, mismos resultados, distinto árbol.
No se ha incorporado todavía FlexFlood al ejecutor común.

## Datos reales / OSM

Este ejecutor acepta JSON de listas o texto TSV con enteros:

- Datos: `x y`, una fila por punto.
- Consultas: `xmin ymin xmax ymax`, límites **inclusivos**.
- Máximo 31 bits por dimensión; dos dimensiones.
- Los snapshots antiguo/nuevo tienen igual cantidad y orden de filas. La fila i
  identifica el mismo objeto; si cambia, se ejecuta una eliminación e inserción.
  No es un lector de archivos `.osm`/`.pbf` ni un generador de cargas de actualización.

Ejemplo, cuando tengas ambos snapshots preparados:

```bash
python adaptive_bmtree/run.py \
  --tree fast_result/uniform_1000000_skew_1000_dim2/mcts/1000_0_3_3_2/0/best_tree.txt \
  --data osm_antes.tsv \
  --new-data osm_despues.tsv \
  --queries consultas_antes.tsv \
  --new-queries consultas_despues.tsv \
  --learn-initial \
  --sample 10000 --rollouts 64 --levels 6 \
  --threshold 0.1 --budget 0.5 --seed 42 \
  --out osm_semilla42
```

`--learn-initial` usa del archivo importado el dominio de bits y aprende un árbol
nuevo desde Z-order solo con consultas de entrenamiento. Evita heredar la fuga
entrenamiento/prueba del script original, que usa todo el workload para entrenar.
Sin esta opción se conserva exactamente el árbol importado y se registra que su
historial de entrenamiento no está verificado. No presentes entonces la evaluación
como prueba independiente si ese árbol ya vio las consultas retenidas.

No trunques latitud/longitud sin definir una transformación. Para OSM necesitas
un dominio fijo y la misma normalización para ambos índices y snapshots. Si
cuantizas, declara su resolución y compara ambos contra la verdad terreno de
esos mismos enteros. La exactitud sobre coordenadas geográficas originales
requiere conservarlas y filtrar con ellas; eso no está implementado aquí.

## Diferencias que debes declarar frente al paper

Esto implementa los mecanismos descritos, pero NO permite afirmar que ya tenemos
una réplica idéntica ni que obtendremos los valores de sus gráficas:

1. Reimplementación independiente de MCTS: UCT con media, horizonte hasta tres
   transiciones por rollout, expansión muestreada limitada a 16 acciones por estado,
   división obligatoria en los primeros dos bits y acciones de división opcional
   hasta seis bits consumidos. Son límites de ingeniería,
   no una copia exacta de todos los hiperparámetros de los autores.
2. Agrupación determinista de consultas en hasta cuatro clusters conjuntos de
   log-área/log-aspecto estandarizados; 30 iteraciones máximas. El paper no fija
   todos esos detalles. Se usa JS con log base 2 y categoría de ausencia para
   histogramas vacíos; el umbral debe interpretarse bajo esta convención.
3. Donde faltan descendientes se completan virtualmente dos bits para medir
   distribución. No se seleccionan regiones dentro de una cadena de hijos
   compartidos, pues su dominio puede ser una unión de cajas; sí puede
   reentrenarse toda la región que contiene esa cadena.
4. Selección incremental de regiones para la regla de mejora <1%, siempre
   disjuntas y dentro del presupuesto estricto. Con presupuesto 0.5 no se
   selecciona una región de exactamente media área.
5. ScanRange reproduce la diferencia de IDs de bloques (no suma 1). Se usa
   asignación por límite inferior de bloque con búsqueda binaria, límites
   saturados y bloque configurable. Es un objetivo aproximado sobre muestra;
   no son lecturas físicas de disco ni nodos visitados del B-tree en RAM.
6. Inicialización, controlador Python y runtime C++ se ejecutan por lotes.
   Faltan conexión persistente online, carga streaming de OSM masivo y comparación
   instrumentada con FlexFlood. El lector actual mantiene snapshots completos en
   memoria; no se ha validado para 100 millones de objetos.

## Protocolo para el paper

Primero ejecutar las pruebas y la demo. Después fijar dataset OSM, precisión,
operaciones y consultas comunes. Separar entrenamiento, validación para elegir
hiperparámetros y prueba final: este ejecutor ofrece entrenamiento/prueba, no un
tercer conjunto de validación. Nunca ajustar parámetros con el conjunto de prueba.
Ejecutar varias semillas y repeticiones de proceso, informar dispersión y equipo.
No seleccionar solo la mejor corrida ni ocultar el tiempo de mantenimiento.
Comparar también un BMTree congelado sobre el mismo snapshot actualizado (la
fase `shifted`) y, si se estudia el costo de adaptación, añadir un reentrenamiento
completo bajo el mismo presupuesto de búsqueda. Estos resultados no están
proporcionados por la demo.

Los valores de `evidence/demo_results.json` solo acreditan funcionamiento en un
entorno de prueba concreto; no son resultados de OSM ni una comparación ganada
contra FlexFlood.
