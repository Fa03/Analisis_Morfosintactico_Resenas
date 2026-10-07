"""Traducción al español con MarianMT, por lotes y con reintentos."""

import logging
import re
import time

log = logging.getLogger("pipeline")


def cargar_traductor(nombre_modelo):
    """Carga el modelo y el tokenizador de MarianMT."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Cargando %s en %s", nombre_modelo, dispositivo)
    tokenizador = MarianTokenizer.from_pretrained(nombre_modelo)
    modelo = MarianMTModel.from_pretrained(nombre_modelo).to(dispositivo)
    modelo.eval()
    return tokenizador, modelo, dispositivo


def dividir_oraciones(texto):
    """Divide un texto en oraciones para no superar el límite del modelo."""
    return [o for o in re.split(r"(?<=[.!?])\s+", texto) if o]


def traducir_lote(textos, tokenizador, modelo, dispositivo, max_reintentos, espera,
                  num_beams=4):
    """Traduce un lote de textos y reintenta si ocurre un error."""
    import torch

    for intento in range(1, max_reintentos + 1):
        try:
            entradas = tokenizador(textos, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512).to(dispositivo)
            with torch.no_grad():
                salida = modelo.generate(**entradas, num_beams=num_beams, max_length=512)
            return tokenizador.batch_decode(salida, skip_special_tokens=True)
        except Exception as error:  # error de memoria, de red, etc.
            log.warning("Error en el lote (intento %d/%d): %s", intento, max_reintentos, error)
            time.sleep(espera * intento)
    return [None] * len(textos)  # si todos los intentos fallan, se marca vacío


def armar_oraciones(df):
    """Crea la lista de (índice de reseña, posición, oración)."""
    oraciones = []
    for indice, texto in df["review_text"].items():
        for posicion, oracion in enumerate(dividir_oraciones(texto)):
            oraciones.append((indice, posicion, oracion))
    return oraciones


def traducir_oraciones(oraciones, config):
    """Traduce todas las oraciones por lotes y devuelve la lista de resultados."""
    tokenizador, modelo, dispositivo = cargar_traductor(config["modelo"])

    # Se ordena por longitud para que cada lote sea más eficiente
    orden = sorted(range(len(oraciones)), key=lambda i: len(oraciones[i][2]))
    resultados = [None] * len(oraciones)
    tamano = config["tamano_lote"]

    for inicio in range(0, len(orden), tamano):
        posiciones = orden[inicio:inicio + tamano]
        textos = [oraciones[i][2] for i in posiciones]
        traducidos = traducir_lote(textos, tokenizador, modelo, dispositivo,
                                   config["max_reintentos"], config["espera_reintento"])
        for i, texto in zip(posiciones, traducidos):
            resultados[i] = texto
        if (inicio // tamano) % 50 == 0:
            log.info("Progreso: %d/%d oraciones", min(inicio + tamano, len(orden)), len(orden))
    return resultados


def reconstruir_resenas(df, oraciones, resultados):
    """Une las oraciones traducidas en el orden original de cada reseña."""
    por_resena = {}
    for (indice, posicion, original), traducida in zip(oraciones, resultados):
        por_resena.setdefault(indice, []).append((posicion, traducida or original))

    df = df.copy()
    df["review_text_es"] = df.index.map(
        lambda i: " ".join(t for _, t in sorted(por_resena.get(i, []))))
    return df


def traducir_resenas(df, config):
    """Ejecuta toda la etapa de traducción."""
    oraciones = armar_oraciones(df)
    log.info("Oraciones a traducir: %d", len(oraciones))
    resultados = traducir_oraciones(oraciones, config)
    fallidas = sum(r is None for r in resultados)
    log.info("Traducción terminada. Oraciones fallidas (quedan en inglés): %d", fallidas)
    return reconstruir_resenas(df, oraciones, resultados)
