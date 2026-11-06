# BMTree v04: correspondencia con el artículo y límites

Fuente principal: PDF adjunto «BMTree: Designing, Learning, and Updating Piecewise
Space-Filling Curves for Multi-Dimensional Data Indexing», secciones IV–VI,
páginas 6–10. Este PDF es la versión ampliada con reentrenamiento parcial, no
solamente el trabajo VLDB 2023 citado como [29]. Se consultaron además los métodos
Node/ZNode y MCTS del repositorio original local Learned-BMTree (commit
230ead0d82489b2f199baf165b0efa5f0c99b8c7).

Esta es una reimplementación independiente del algoritmo para puntos 2D enteros
(hasta 31 bits por coordenada) y consultas de rango exactas. NO es una certificación
de reproducción idéntica ni una réplica de los experimentos PostgreSQL/RSMI.
El almacenamiento es cpp-btree en RAM, como requiere esta comparación sin BD.

## Cambios comprobables

| Parte del paper | v04 | Ubicación |
|---|---|---|
| IV: bit más significativo pendiente, división o hijo compartido | Conservado; se comprobó contra Node/ZNode original | encode, C++ Tree::key |
| V: estado = árbol parcial, acción conjunta dimensión/división | Ambos tipos disponibles en cada nodo; eliminadas divisiones forzadas de v03 | choices, fill_one |
| V: construir nivel por nivel | Sí; nivel entendido como una transición de toda la frontera | train_forest |
| V: GAS | Evalúa cada bit válido por ScanRange y compromete la mejor decisión secuencialmente en la frontera | greedy_action |
| V: simulación con continuación Z | Sí, sin ejecutar consultas reales para evaluar recompensas | encode, cost |
| V: SR = IDmax − IDmin | Conservado; bloques de muestra ordenada, extremos con searchsorted | scan_ranges |
| V: recompensa normalizada frente a Z | (SR_Z − SR_T)/SR_Z; usa medias, equivalente a cociente de sumas sobre el mismo workload | train_forest |
| V: backpropagation por máximo | V=max(V,reward), nunca promedio acumulativo | Policy.observe |
| V: UCT y elegir acción con mayor valor | Expresión impresa V/visitas + sqrt(log(visitas_padre)/visitas); compromiso por mayor V | uct, train_forest |
| VI-A: JS de datos y consultas, dos niveles, alpha=0.5 | Sí; descendientes reales, centros de consulta, grupos por área/aspecto | cells, query_clusters, detect |
| VI-B / Alg.1: BFS, filtro de shift, prioridad OP, presupuesto estricto | Sí; no descender si el padre no pasa, ordenar OP por nivel | detect, select |
| VI-C / Alg.2: borrar y entrenar juntos los subárboles elegidos | Sí; primera ronda usa todo el conjunto seleccionado | adapt |
| VI-D: más regiones si mejora menor de 1% | Vuelve a detectar sobre el árbol nuevo y usa regiones no entrenadas dentro del presupuesto acumulado | adapt |
| VI: conservar resto de la curva y recodificar solo regiones afectadas | Sí; incluye regiones que son uniones de franjas después de hijos compartidos | Region.domains, BMIndex::rekey |

No queda la ausencia de GAS ni el backup medio que existían en v03. Se quitó
el horizonte interno de 3 transiciones, el límite de 16 acciones por estado y
el límite obligatorio de 8 rondas. levels y rollouts son presupuestos explícitos.
El conjunto exponencial de acciones se explora de forma perezosa, sin construirlo
entero ni recortarlo: primero GAS, luego acciones inéditas, finalmente UCT cuando
se agotan las acciones de un estado. La búsqueda tiene presupuesto finito y no
garantiza la mejor curva global.

## Decisiones y ambigüedades que no se deben ocultar

1. La ecuación UCT impresa divide V por visitas a pesar de definir V como máximo.
   El mcts.py original guarda una cantidad cuya división por visitas representa
   el máximo observado (con peculiaridades de inicialización). La v04 sigue
   literalmente la ecuación y el máximo del PDF adjunto. No afirmamos igualdad
   de trayectorias de búsqueda con ese repositorio.
2. GAS no detalla cómo desempatar ni cómo combinar su elección de bits con split.
   Se prueban dimensión y split, con empate por menor dimensión y luego split.
   Las demás acciones siguen disponibles para exploración. No se fuerza split.
3. El PDF no fija completamente clustering, número de grupos, tratamiento de
   vacíos y base logarítmica. Usamos k=4, agrupamiento determinista conjunto de
   consultas antiguas/nuevas en cada subespacio, log(área) y log(aspecto)
   estandarizados, JS en base 2 y categoría de ausencia cuando no hay consultas.
   Área corresponde a puntos de retícula inclusiva. Son decisiones explícitas.
4. Los hijos compartidos mantienen el dominio del padre. Sus descendientes pueden
   ocupar varias franjas. Se representan como uniones de cajas disjuntas, no como
   una única caja que incluiría puntos ajenos. Las cajas se exportan al C++.
5. Alg.1 solo encola hijos si el padre pasa el umbral. Se sigue esa regla literal.
   Un cambio dentro de un cuadrante puede conservar todos los conteos de los
   nietos de la raíz y quedar sin detectar. La demo actual puede NO reentrenar.
   No se debe bajar el umbral buscando mejores números; se declara como parámetro
   y se estudia su sensibilidad por separado.
6. Al seleccionar regiones, se omiten descendientes de una región ya seleccionada:
   no sumamos dos veces su área ni reentrenamos dos veces el mismo subárbol.
   El presupuesto acumulado es estricto: con 0.5 no se admite exactamente 0.5.
   La instrucción recursiva de Alg.2 no precisa por completo el presupuesto entre
   rondas: se mantiene acumulativo, con nuevas regiones y sin repetir anteriores.
7. El coste se calcula solo con consultas cuyos centros caen en regiones elegidas,
   pero los límites de consulta pueden estar fuera de ellas. Se conserva el
   orden global del dataset muestreado y se recalculan los extremos afectados.
8. Si SR_Z=0 se usa normalizador 1 para evitar división por cero. Los tiempos de
   Python y C++ se miden por componentes. La memoria sigue siendo estimación
   estructural residente, con registros, no RSS ni pico de entrenamiento.

## Optimización conservando el algoritmo

- Caché LRU de hasta 4096 costes por entrenamiento.
- Reutilización de claves de puntos y extremos fuera de las regiones modificadas.
- Cálculo vectorizado, búsqueda binaria de bloques y generación perezosa de acciones.
- Reutilización de estados visitados al comprometer el siguiente nivel.
- La validación con fuerza bruta permanece fuera del tiempo de consulta.

No se han elegido parámetros mirando las 60 consultas de prueba. No se promete
que v04 gane a FlexFlood o mejore todas las métricas de v03; un algoritmo de
aprendizaje distinto puede generar curvas y tiempos distintos.

## Validación

- 12 pruebas unitarias/integración: GAS contra evaluación separada por bit,
  máximo/UCT, búsqueda de 5 niveles, regiones compartidas, ausencia de cambio,
  cambio real que sí dispara entrenamiento, bosque inicial conjunto, límites,
  duplicados, ScanRange, recodificación C++ y conservación de claves exteriores.
- Auditoría independiente: 25.920 consultas antes y 25.920 después de actualizar
  y recodificar, sobre 20 pares de árboles. Oráculo Python independiente posterior.
- Caso adicional compartido: 1.296 consultas nativas posteriores comparadas con
  fuerza bruta Python; el ejecutable también valida antes y después internamente.
- 6.400 claves en 100 árboles coinciden con Node/ZNode del repositorio original.
  Esto compara codificación, NO equivalencia del entrenamiento.
- Demos completas con 10.000 puntos y ambos motores. Prueba explícita con umbral
  cero para ejercitar reentrenamiento extremo a extremo: es una prueba funcional,
  no una elección de parámetros recomendada ni evidencia de mejora de rendimiento.

Las evidencias están en evidence/. No constituyen una prueba formal universal.
Falta evaluar datasets reales, varias semillas y tamaños, y documentar parámetros
para un estudio experimental publicable. La extensión a más de dos dimensiones
no forma parte de esta entrega 2D; no debe afirmarse que aquí se reprodujeron
los experimentos multidimensionales de 2 a 6 dimensiones del artículo.
