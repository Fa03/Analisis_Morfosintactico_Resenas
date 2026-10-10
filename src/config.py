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
