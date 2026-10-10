"""Muestreo estratificado por género y año."""

import logging

import pandas as pd

log = logging.getLogger("pipeline")


def muestrear_estratificado(df, total, semilla):
    """Toma el mismo número de reseñas por género y por año."""
    num_generos = df["main_genre"].nunique()
    num_anios = df["year"].nunique()
    por_grupo = total // (num_generos * num_anios)

    partes = []
    for _, grupo in df.groupby(["main_genre", "year"]):
        partes.append(grupo.sample(n=min(por_grupo, len(grupo)), random_state=semilla))
    muestra = pd.concat(partes)

    # Si algún grupo no alcanzó la cuota, se completa con reseñas restantes
    faltantes = total - len(muestra)
    if faltantes > 0:
        log.warning("Faltan %d reseñas, se completan con las restantes.", faltantes)
        restantes = df.drop(muestra.index)
        extra = restantes.sample(n=min(faltantes, len(restantes)), random_state=semilla)
        muestra = pd.concat([muestra, extra])

    muestra = muestra.sample(frac=1, random_state=semilla).reset_index(drop=True)
    log.info("Muestra final: %d reseñas", len(muestra))
    log.info("Distribución género x año:\n%s",
             pd.crosstab(muestra["main_genre"], muestra["year"]).to_string())
    return muestra
