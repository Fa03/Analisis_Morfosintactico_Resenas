"""Programa principal: ejecuta el pipeline completo.

Uso:
    pip install pandas transformers torch sentencepiece
    python principal.py
"""

import logging
from pathlib import Path

from carga import cargar_datos
from config import CONFIG
from filtros import filtrar_generos, filtrar_longitud, filtrar_periodos
from muestreo import muestrear_estratificado
from traduccion_1 import traducir_resenas

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
    salida_procesado = Path(config["salida_procesado"])
    salida_procesado.parent.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(salida_procesado, index=False, encoding="utf-8")
    logging.info("Archivo guardado: %s", salida_procesado)

    if not config.get("traducir"):
        logging.info("Traducción omitida (config['traducir']=False). "
                     "Ejecutar src/traducir.py, p. ej. en Google Colab.")
        return

    traducido = traducir_resenas(muestra, config)
    salida_traducido = Path(config["salida_traducido"])
    salida_traducido.parent.mkdir(parents=True, exist_ok=True)
    traducido.to_csv(salida_traducido, index=False, encoding="utf-8")
    logging.info("Archivo guardado: %s", salida_traducido)


if __name__ == "__main__":
    principal()
