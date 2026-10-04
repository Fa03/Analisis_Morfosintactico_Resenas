"""Configuración central del pipeline."""

CONFIG = {
    # Rutas locales (dataset rmisra/imdb-spoiler-dataset)
    "ruta_resenas": "data/IMDB_reviews.json",
    "ruta_peliculas": "data/IMDB_movie_details.json",

    # Segundo dataset opcional (thedevastator/reviewsof-imdb-movies).
    # El diccionario renombra sus columnas a los nombres del pipeline.
    "ruta_extra": None,  # ejemplo: "data/imdb_reviews.csv"
    "columnas_extra": {},  # ejemplo: {"Review": "review_text", "Movie": "movie_id"}

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

    # Traducción
    "modelo": "Helsinki-NLP/opus-mt-en-es",
    "tamano_lote": 16,
    "max_reintentos": 3,
    "espera_reintento": 5,  # segundos, se multiplica en cada reintento

    # Salidas
    "salida_procesado": "processed.csv",
    "salida_traducido": "resenas_es.csv",
}
