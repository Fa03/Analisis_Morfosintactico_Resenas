# Análisis morfosintáctico de reseñas de IMDb

Este proyecto aplica técnicas de minería de textos sobre reseñas de películas. El flujo carga los datos, los filtra, toma una muestra equilibrada, los traduce al español, los etiqueta con dos analizadores gramaticales y calcula métricas morfosintácticas comparando grupos de reseñas.

## Objetivo

Comparar el estilo de las reseñas según su género, su polaridad, su año y la presencia de spoilers, usando métricas como la adjetivación, la proporción verbo/sustantivo, la intensidad y la longitud. Además, contrasta las etiquetas del etiquetador spaCy con las de NLTK.

## Estructura del repositorio

```
POS_Tagging/
├── data/
│   ├── raw/            # Datasets originales de IMDb (no versionados)
│   ├── processed/      # Datos limpios, traducidos y listos para etiquetar
│   └── results/        # Etiquetas POS generadas
├── notebooks/          # Cuaderno auxiliar de traducción
├── outputs/            # Figuras y caché del análisis
│   └── figuras/
├── src/                # Código del pipeline
├── requeriments.txt    # Dependencias
├── USO_DE_IA.md        # Registro del uso de asistentes de IA
└── README.md
```

## Requisitos e instalación

- Python 3.13 dentro del entorno virtual `.venv`.
- Las dependencias se instalan desde `requeriments.txt` (ese es el nombre real del archivo).
- Se descargan aparte los recursos que no vienen en el archivo anterior:
  - Modelo de spaCy: `es_core_news_md`.
  - Corpora de NLTK: `cess_esp` (etiquetado en español) y `punkt` (segmentación de oraciones).

En Windows el intérprete se invoca como `.venv\Scripts\python.exe`; en Linux, como `.venv/bin/python`.

## Flujo del pipeline

El proyecto se ejecuta por etapas, en este orden:

| Etapa | Script | Entrada | Salida |
|-------|--------|---------|--------|
| Carga y unión | `src/carga.py` | Datasets crudos de IMDb | Reseñas unidas con metadatos |
| Filtrado | `src/filtros.py` | Reseñas unidas | Reseñas limpias por longitud, año y género |
| Muestreo | `src/muestreo.py` | Reseñas filtradas | Muestra equilibrada |
| Orquestación | `src/principal.py` | Todo lo anterior | `data/processed/processed.csv` |
| Traducción | `src/traducir.py` + `src/traductor.py` | `processed.csv` | `processed_es.csv` |
| Limpieza final | `src/LimpiezaResenasEspanol.py` | `processed_es.csv` | Corpus depurado |
| Etiquetado POS | `src/pos_tagging.py` | Corpus depurado | `data/results/pos_analysis*.csv` |
| Análisis de métricas | `src/metrics.py` | Etiquetas y metadatos | Tablas, figuras y caché |

## Descripción de los scripts

- `carga.py`: lee los JSON de reseñas y de películas, los une y crea la columna de año.
- `filtros.py`: quita HTML, limita la longitud de las reseñas y conserva los períodos y géneros con más volumen.
- `muestreo.py`: toma el mismo número de reseñas por género y año, con semilla fija para reproducibilidad.
- `principal.py`: encadena carga, filtrado y muestreo; opcionalmente dispara la traducción.
- `traductor.py`: traduce inglés a español con `Helsinki-NLP/opus-mt-en-es`; agrupa oraciones para acelerar el proceso.
- `traducir.py`: interfaz de línea de comandos de la traducción; guarda por bloques y es reanudable.
- `LimpiezaResenasEspanol.py`: normaliza el corpus traducido antes del etiquetado.
- `pos_tagging.py`: etiqueta cada token con spaCy y NLTK y guarda los resultados en CSV.
- `metrics.py`: calcula las métricas por grupo y la comparación entre etiquetadores; genera figuras y una caché.

## Configuración

La configuración vive en `src/config.py`, con dos diccionarios:

- `CONFIG`: rutas de entrada y salida, parámetros de filtrado y muestreo, y opciones de traducción.
- `METRICAS`: parámetros del análisis morfosintáctico (umbrales de polaridad, base de normalización, géneros comparados, léxico de negaciones e intensificadores y etiquetas excluidas).

Parámetros principales del muestreo: reseñas de 30 a 400 palabras, dos períodos, cinco géneros principales y una muestra total de 3000 reseñas (semilla 42).

## Métricas y decisiones de análisis

El análisis se organiza en cuatro ejes y una comparación entre etiquetadores:

- **Eje A — Adjetivación y estilo.** Compara Comedy y Drama en adjetivos por cada 100 tokens y en la proporción verbo/sustantivo por género.
- **Eje B — Intensidad y polaridad.** Mide adverbios de intensidad según polaridad y extrae los adjetivos más distintivos de las reseñas positivas y negativas.
- **Eje C — Evolución temporal.** Estudia la simplicidad gramatical y la longitud media de las reseñas entre años.
- **Eje D — Spoilers.** Compara el uso de verbos y adjetivos entre reseñas con y sin spoilers.
- **Comparación NLTK vs spaCy.** Mide la concordancia global, clasifica las discrepancias por causa y muestra una matriz de confusión.

Decisiones fijadas:

- Las métricas se normalizan por cada 100 tokens.
- Se excluye la puntuación y los símbolos de todos los cálculos.
- La diversidad léxica (TTR) se calcula sobre los primeros 100 lemas de cada reseña.
- Se consideran negativas las reseñas con rating menor o igual a 4 y positivas las de rating mayor o igual a 7; las neutras quedan fuera.
- El Eje A compara Comedy contra Drama.

## Ejecución

Etapa de preparación:

```
.venv\Scripts\python.exe src\principal.py
.venv\Scripts\python.exe src\traducir.py
.venv\Scripts\python.exe src\LimpiezaResenasEspanol.py
.venv\Scripts\python.exe src\pos_tagging.py
```

Análisis de métricas:

```
.venv\Scripts\python.exe src\metrics.py
```

El análisis imprime las tablas y las interpretaciones en la consola y escribe las figuras en `outputs/figuras/`. En la primera ejecución reejecuta spaCy para obtener lemas y oraciones, y guarda el resultado en `outputs/metrics_tokens.csv`; en ejecuciones posteriores reutiliza esa caché.

## Salidas y resultados

Figuras generadas en `outputs/figuras/`:

| Archivo | Contenido |
|---------|-----------|
| `ejeA_adjetivacion_comedy_drama.png` | Adjetivos por 100 tokens: Comedy vs Drama |
| `ejeA_ratio_verbo_sustantivo.png` | Proporción verbo/sustantivo por género |
| `ejeB_intensidad_polaridad.png` | Adverbios de intensidad por polaridad |
| `ejeB_adjetivos_distintivos.png` | Adjetivos más distintivos por polaridad |
| `ejeC_simplicidad_gramatical.png` | Simplicidad gramatical por año |
| `ejeC_longitud_resenas.png` | Longitud media de las reseñas por año |
| `ejeD_spoilers_estilo.png` | Verbos y adjetivos según spoiler |
| `nltk_concordancia_upos.png` | Concordancia de etiquetas NLTK vs spaCy |
| `nltk_matriz_confusion.png` | Matriz de confusión de etiquetas |

Resultados de referencia sobre el corpus actual (3000 reseñas, años 2006 y 2017, géneros Crime, Adventure, Action, Drama y Comedy):

- Adjetivación: Comedy 8.67 frente a Drama 8.16 adjetivos por 100 tokens.
- Proporción verbo/sustantivo: Drama 0.576 (la más alta) frente a Action 0.519 (la más baja).
- Intensidad: 0.385 intensificadores por 100 tokens en reseñas negativas frente a 0.465 en positivas.
- Longitud media: baja de 193.4 a 167.1 tokens entre 2006 y 2017.
- Concordancia global entre etiquetadores, sin puntuación: 82.93%.

## Limitaciones

- El corpus cubre solo dos años (2006 y 2017) y cinco géneros; no existen las categorías "Terror" ni "Comedia".
- Los archivos POS no incluyen lemas, morfología ni oraciones, por lo que `metrics.py` reejecuta spaCy para obtenerlos.
- NLTK usa el corpus `cess_esp`, entrenado sobre noticias, lo que explica parte del desacuerdo con spaCy.
- La polaridad depende del rating y deja fuera las reseñas neutras.

## Inconsistencias y elementos a corregir

Los siguientes puntos se documentan como pendientes y no afectan al análisis actual:

1. `src/principal.py` importa el módulo `traduccion_1`, que no existe en `src/`. El script no es ejecutable tal cual.
2. `src/traducir.py` genera por defecto `<entrada>_es.csv` (`processed_es.csv`), mientras que `config.py` define `salida_traducido` como `resenas_es.csv`. Conviene unificar el nombre.
3. El `.gitignore` excluye `/src/LimpiezaResenasEspanol.py`, aunque ese script forma parte del pipeline y el resto del código lo referencia.
4. `outputs/` no está en `.gitignore`; hay que decidir si las figuras y la caché se versionan o no.
5. `requeriments.txt` no declara la descarga de `punkt` ni de `cess_esp` de NLTK, necesarios para la segmentación y el etiquetado.
6. El nombre `requeriments.txt` tiene una errata; lo habitual es `requirements.txt`.
7. `config.py` conserva opciones de traducción antiguas (`modelo`, `tamano_lote`, `max_reintentos`, `espera_reintento`) que `traductor.py` ya no utiliza.

## Uso de IA

El registro de consultas a asistentes de IA y los aprendizajes asociados se documentan en `USO_DE_IA.md`.

## Autoría

Proyecto de minería de textos elaborado por Fabián Brenes Loría para el curso de Minería de Textos. Fecha de entrega: 14 de octubre de 2026.
