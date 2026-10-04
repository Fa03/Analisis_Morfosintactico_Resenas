"""Limpieza y filtros: longitud, períodos y géneros."""

import logging
import re

log = logging.getLogger("pipeline")


def limpiar_texto(texto):
    """Quita etiquetas HTML y espacios repetidos."""
    texto = re.sub(r"<[^>]+>", " ", str(texto))
    return re.sub(r"\s+", " ", texto).strip()


def filtrar_longitud(df, min_palabras, max_palabras):
    """Conserva las reseñas con una cantidad de palabras dentro del rango."""
    df = df.copy()
    df["review_text"] = df["review_text"].apply(limpiar_texto)
    df["n_words"] = df["review_text"].str.split().str.len()
    df = df[df["n_words"].between(min_palabras, max_palabras)]
    log.info("Tras filtro de longitud (%d-%d palabras): %d", min_palabras, max_palabras, len(df))
    log.info("Palabras por reseña:\n%s", df["n_words"].describe().round(1).to_string())
    return df


def filtrar_periodos(df, num_periodos, min_resenas):
    """Selecciona los años con más reseñas (mínimo min_resenas cada uno)."""
    conteo = df["year"].value_counts()
    log.info("Top 10 años por volumen:\n%s", conteo.head(10).to_string())

    validos = conteo[conteo >= min_resenas]
    if len(validos) < num_periodos:
        raise ValueError(f"Solo {len(validos)} años tienen {min_resenas}+ reseñas.")

    anios = sorted(validos.head(num_periodos).index.tolist())
    df = df[df["year"].isin(anios)]
    log.info("Años seleccionados: %s | Registros: %d", anios, len(df))
    return df, anios


def filtrar_generos(df, num_generos, min_resenas):
    """Conserva los géneros principales y asigna un solo género por reseña."""
    conteo = df.explode("genre")["genre"].value_counts()
    log.info("Top 10 géneros en el subconjunto:\n%s", conteo.head(10).to_string())

    validos = conteo[conteo >= min_resenas]
    if len(validos) < num_generos:
        raise ValueError(f"Solo {len(validos)} géneros tienen {min_resenas}+ reseñas.")
    principales = validos.head(num_generos)

    # Cada reseña se asigna a su género más raro entre los principales, para balancear
    def elegir_genero(generos):
        opciones = [g for g in generos if g in principales.index]
        return min(opciones, key=lambda g: principales[g]) if opciones else None

    df = df.copy()
    df["main_genre"] = df["genre"].apply(elegir_genero)
    df = df.dropna(subset=["main_genre"])
    log.info("Géneros seleccionados: %s", principales.index.tolist())
    log.info("Registros por género:\n%s", df["main_genre"].value_counts().to_string())
    return df, principales.index.tolist()
