"""Programa principal: ejecuta el pipeline completo.

Uso:
    pip install pandas transformers torch sentencepiece
    python principal.py
"""

import logging

from carga import cargar_datos
from config import CONFIG
from filtros import filtrar_generos, filtrar_longitud, filtrar_periodos
from muestreo import muestrear_estratificado
from traduccion import traducir_resenas

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def preparar_muestra(config):
    """Carga, filtra y muestrea los datos."""
    df = cargar_datos(config)
    df = filtrar_longitud(df, config["min_palabras"], config["max_palabras"])
    df, _ = filtrar_periodos(df, config["num_periodos"], config["min_resenas_periodo"])
    df, _ = filtrar_generos(df, config["num_generos"], config["min_resenas_genero"])
    return muestrear_estratificado(df, config["total_muestra"], config["semilla"])


def principal(config=CONFIG):
    muestra = preparar_muestra(config)

    # La columna de géneros es una lista, por eso se guarda como texto
    muestra["genre"] = muestra["genre"].apply(lambda g: "|".join(g))
    muestra.to_csv(config["salida_procesado"], index=False, encoding="utf-8")
    logging.info("Archivo guardado: %s", config["salida_procesado"])

    traducido = traducir_resenas(muestra, config)
    traducido.to_csv(config["salida_traducido"], index=False, encoding="utf-8")
    logging.info("Archivo guardado: %s", config["salida_traducido"])


if __name__ == "__main__":
    principal()
