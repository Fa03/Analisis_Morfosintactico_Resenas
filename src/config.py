"""Configuración central del pipeline."""

from pathlib import Path

# Raíz del proyecto: src/config.py -> padre de src/
BASE_DIR = Path(__file__).resolve().parents[1]


CONFIG = {
    # Rutas locales (dataset rmisra/imdb-spoiler-dataset)

    "ruta_resenas": BASE_DIR / "data/raw/IMDB Spoiler Dataset/IMDB_reviews.json",
    "ruta_peliculas": BASE_DIR / "data/raw/IMDB Spoiler Dataset/IMDB_movie_details.json",

    # Segundo dataset opcional (data/raw/Reviews_IMDB/reviews.csv).
    # El diccionario renombra sus columnas a los nombres del pipeline.
    "ruta_extra": None,  # ejemplo: BASE_DIR / "data/raw/Reviews_IMDB/reviews.csv"
    "columnas_extra": {},  # ejemplo: {"text": "review_text", "movieId": "movie_id", "date": "review_date"}

    # Filtros
    "min_palabras": 30,
    "max_palabras": 400,
    "fuente_anio": "review_date",  # "review_date" o "release_date"
    "num_periodos": 2,
    "min_resenas_periodo": 1000,
    "num_generos": 5,
    "min_resenas_genero": 400,

    # Muestreo
    "total_muestra": 3000,
    "semilla": 42,

    # Traducción (se ejecuta con src/traducir.py, p. ej. en Google Colab/GPU)
    "traducir": False,
    "modelo": "Helsinki-NLP/opus-mt-en-es",
    "tamano_lote": 16,
    "max_reintentos": 3,
    "espera_reintento": 5,  # segundos, se multiplica en cada reintento

    # Salidas
    "salida_procesado": BASE_DIR / "data/processed/processed.csv",
    "salida_traducido": BASE_DIR / "data/processed/resenas_es.csv",
}


# Configuración del análisis morfosintáctico (src/metrics.py).
METRICAS = {
    # Rutas de entrada y salida
    "ruta_tokens": BASE_DIR / "data/results/pos_analysis_2.csv",
    "ruta_metadatos": BASE_DIR / "data/processed/processed_es.csv",
    "carpeta_salida": BASE_DIR / "outputs",
    "carpeta_figuras": BASE_DIR / "outputs/figuras",
    "ruta_cache": BASE_DIR / "outputs/metrics_tokens.csv",

    # Modelo de spaCy y parámetros de cálculo
    "modelo_spacy": "es_core_news_md",
    "base_normalizacion": 100,
    "tamano_bloque": 100,

    # Umbrales de polaridad a partir del rating (1-10)
    "rating_negativo": 4,
    "rating_positivo": 7,

    # Géneros comparados en el Eje A (proxy de Comedia vs Terror)
    "generos_eje_a": ["Comedy", "Drama"],

    # Etiquetas que se excluyen de todas las métricas
    "etiquetas_excluidas": ["PUNCT", "SYM", "SPACE", "X"],

    # Léxico usado en las métricas del Eje B
    "negaciones": [
        "no", "nunca", "jamás", "jamas", "nada", "nadie", "tampoco",
        "ni", "sin", "ningún", "ningun", "ninguna", "ninguno", "sino",
    ],
    "adverbios_intensidad": ["muy", "demasiado", "totalmente"],
}
