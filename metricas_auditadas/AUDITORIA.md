# Auditoría v03: exactitud no equivale a fidelidad al paper

## Dictamen

El Recall micro se calcula a partir de respuestas reales comparadas con fuerza
bruta; no está fijado en 100. La versión v02 sólo publica la tabla cuando todos
los resultados son exactos. Por ese diseño, toda tabla final publicada debe
mostrar Recall 100%; los casos incorrectos se guardan en JSON crudo y abortan.

Esto valida únicamente las consultas ejecutadas. No demuestra ausencia universal
de errores, fidelidad del aprendizaje al paper ni resultados experimentales
comparables a los publicados por los autores.

## Dataset real de esta demo

El ejecutor `run.py: make_workload` genera un dataset NUEVO. No carga el archivo
uniform_1000000.json del repo ni usa data/generator.cpp de FlexFlood. Por defecto:
10.000 puntos 2D únicos, coordenadas enteras de 20 bits; inicia con uniforme.
Sustituye puntos del cuadrante inferior izquierdo por puntos uniformes en una
franja más estrecha. Es un cambio localizado sintético, no una gaussiana ni OSM.
240 consultas entrenan ambos índices y 60 diferentes sirven de prueba. Las
repeticiones del cronómetro no aumentan el número de consultas distintas.

## Cómo se computa y se pone a prueba Recall

En common.hpp, validate enumera resultados del índice, recorre todos los puntos
para formar la respuesta correcta, ordena ambas listas y cuenta su intersección.
Recall micro = suma TP / suma esperados. También comprueba igualdad de listas y
cuenta falsos positivos. Si todo el workload tiene cero resultados esperados,
se define Recall=1 por convenio, pero se sigue exigiendo igualdad exacta.

Se inyectaron fallos deliberados en un índice simulado:
- Recuperar uno de dos puntos: Recall=0.5 y consulta marcada incorrecta.
- Recuperar los dos y agregar un punto falso: Recall=1, FP=1 e incorrecta.
- Recuperar sólo un punto falso: Recall=0, FP=1 e incorrecta.
- Respuesta y verdad terreno vacías: Recall=1 por convenio y correcta.

La v03 añade TP, esperados, FN, FP, cantidad de consultas y consultas vacías tanto
al JSON como al resumen. No se acepta Recall 100 como criterio suficiente.

## Resultados de la revisión

Tres demos de 10.000 puntos con la versión corregida, 60 consultas distintas por
semilla; los contadores siguientes son por cada índice, después del mantenimiento:

| Semilla | TP | Esperados | FN | FP | Consultas vacías |
|---|---:|---:|---:|---:|---:|
| 42 | 4106 | 4106 | 0 | 0 | 9 |
| 43 | 4227 | 4227 | 0 | 0 | 11 |
| 7 | 4350 | 4350 | 0 | 0 | 8 |

Son resultados reproducidos aquí; los registros guardados en tu PC siguen siendo
la fuente para tu corrida particular. Las apariciones de un punto en consultas
distintas se cuentan por separado en estas sumas.

Otras comprobaciones:
- 20 pares de árboles con bifurcaciones y ramas compartidas, dominio 8x8.
- Las 1296 cajas de extremos enteros posibles dentro de ese dominio por par:
  25.920 consultas antes de updates y 25.920 después de updates + recodificación.
- Las respuestas C++ posteriores se compararon con un oráculo separado en Python.
- Validación de claves residentes contra el árbol nuevo tras recodificar.
- Biyectividad y monotonicidad de las codificaciones de ambos árboles por caso.
- Conservación de claves fuera del subárbol izquierdo modificado.
- 6.400 claves de 100 árboles comparadas con los métodos Node/ZNode originales,
  extraídos sin modificar mediante AST: coincidencia. Esto NO compara MCTS.
- 256 centros enteros/semienteros comprobados contra la partición corregida.
- 9 pruebas del paquete adaptativo anterior pasaron; una prueba que necesita
  PyTorch se omitió porque no está instalado en este entorno de auditoría.
  La comparación AST anterior cubre codificación original de forma independiente.
- Las dos pruebas de integración de los motores pasaron (límites, vacíos,
  actualizaciones y rechazo de coordenadas duplicadas).

No se trata de una prueba formal para todas las dimensiones, geometrías o datos.

## Fallo encontrado y corregido

Region.mask usaba lo <= centro <= hi. Los extremos de las regiones describen
coordenadas ENTERAS, pero los centros de consultas pueden ser semienteros.
Ejemplo: región izquierda 0..3, derecha 4..7, centro 3.5: ninguna lo aceptaba.

Ahora la pertenencia geométrica usa [lo, hi+1). Para puntos enteros no cambia el
resultado; para centros semienteros elimina los huecos. Esto corrige la asignación
de consultas usada por detección y entrenamiento parcial. Las consultas de
usuario siguen siendo inclusivas en ambos extremos. No se ha atribuido este fallo
a tu resultado concreto sin evidencia; se reprodujo mediante una prueba dirigida.

## Diferencias materiales con BMTree del paper ampliado

La sección V del PDF proporcionado describe Greedy Action Selection (GAS) y
actualización de valores mediante máximo. Nuestra train_forest no implementa
esa GAS: expande acciones aleatorias acotadas y usa acumulación/media de retornos.
Es una variante experimental de MCTS, no el algoritmo de aprendizaje completo.

Además conserva límites propios de ingeniería: horizonte de tres transiciones,
16 acciones por estado, división obligatoria en los primeros dos bits, selección
limitada de subregiones bajo ramas compartidas, clustering concreto de consultas
y un procedimiento incremental de regiones. La región inicial de reentrenamiento
y las rondas adicionales no replican exactamente los algoritmos 1–2.

Todo esto puede cambiar el árbol, ScanRange, costo de entrenamiento y rendimiento.
El Recall puede seguir siendo 100% porque la búsqueda exacta recorre y filtra el
intervalo de claves correcto incluso cuando la curva aprendida es poco eficiente.

La v03 conserva estas diferencias; NO las oculta ni dice haberlas corregido.
Marca la columna BMTree* y explica su carácter experimental. Para afirmar que
se reproduce BMTree del artículo falta implementar/validar su aprendizaje y
alinear la selección de regiones, los hiperparámetros y el protocolo experimental.

## Alcance de las métricas restantes

- El tiempo de adaptación BMTree es suma de componentes Python/C++, no una
  medición continua de un servicio en línea; FlexFlood adapta dentro de updates.
- La memoria es estimación de estructuras residentes con datos propios, no RSS
  ni memoria extra excluyendo payload. BMTree conserva registros borrados e IDs.
- La degradación mezcla efecto de datos/selectividad y adaptación; no es el
  beneficio aislado de adaptar.
- Presupuestos de aprendizaje reducidos para demo; no son los de los papers.
- Un proceso por índice, un orden fijo, pocas consultas distintas y datos 2D.

Por ello las tablas actuales son pruebas funcionales y mediciones del prototipo.
No deben presentarse todavía como resultados de una réplica fiel de BMTree contra
FlexFlood ni como conclusión de superioridad general de un algoritmo.

## Ejecutar la revisión

Extrae la carpeta metricas_auditadas junto a metricas_comunes. Desde el repo:

```bash
source ~/venvs/bmtree/bin/activate
python metricas_auditadas/run.py --out auditoria_metricas_01
python metricas_auditadas/audit_independent.py
python metricas_auditadas/audit_original_encoder.py bmtree/tree.py
```

La demo usa las dependencias ya instaladas. La auditoría independiente compila un
programa C++ y requiere g++; la comparación AST no requiere PyTorch. Los scripts
y evidencias están incluidos en este paquete. Las tablas de rendimiento de las
corridas de auditoría no se usan para elegir parámetros ni proclamar mejoras.
