"""Carga, unión y parseo de los datos."""

import ast
import logging

import pandas as pd

log = logging.getLogger("pipeline")

# Columnas de películas que se conservan del dataset
COLUMNAS_PELICULAS = ["movie_id", "genre", "release_date", "rating"]


def convertir_generos(valor):
    """Convierte el campo de géneros en una lista de Python."""
    if isinstance(valor, list):
        return valor
    if isinstance(valor, str):
        try:
            return ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            return [g.strip() for g in valor.split(",") if g.strip()]
    return []


def cargar_resenas(ruta, ruta_extra=None, columnas_extra=None):
    """Carga las reseñas y agrega el segundo dataset si existe."""
    resenas = pd.read_json(ruta, lines=True)
    log.info("Reseñas cargadas: %d", len(resenas))

    if ruta_extra:
        extra = pd.read_csv(ruta_extra).rename(columns=columnas_extra)
        resenas = pd.concat([resenas, extra], ignore_index=True, sort=False)
        log.info("Reseñas tras agregar el segundo dataset: %d", len(resenas))
    return resenas


def cargar_peliculas(ruta):
    """Carga los metadatos de las películas con las columnas necesarias."""
    peliculas = pd.read_json(ruta, lines=True)[COLUMNAS_PELICULAS]
    # Se renombra para no chocar con el rating de las reseñas
    peliculas = peliculas.rename(columns={"rating": "movie_rating"})
    log.info("Películas cargadas: %d", len(peliculas))
    return peliculas


def unir_datos(resenas, peliculas, fuente_anio):
    """Une reseñas y películas, parsea fechas y crea la columna del año."""
    df = resenas.merge(peliculas, on="movie_id", how="inner")
    log.info("Registros tras el merge: %d", len(df))

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["genre"] = df["genre"].apply(convertir_generos)

    df["year"] = df[fuente_anio].dt.year
    df = df.dropna(subset=["year", "review_text"])
    df["year"] = df["year"].astype(int)
    log.info("Registros con año y texto válidos: %d", len(df))
    return df


def cargar_datos(config):
    """Ejecuta toda la etapa de carga."""
    resenas = cargar_resenas(config["ruta_resenas"], config["ruta_extra"], config["columnas_extra"])
    peliculas = cargar_peliculas(config["ruta_peliculas"])
    return unir_datos(resenas, peliculas, config["fuente_anio"])
