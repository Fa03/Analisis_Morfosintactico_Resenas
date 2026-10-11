"""

Flujo general de ejecución
--------------------------
1. ``cargar_textos`` lee el CSV y valida la columna de texto.
2. ``cargar_modelo_spacy`` carga el modelo en español.
3. ``etiquetar_con_spacy`` tokeniza y etiqueta con spaCy.
4. ``entrenar_tagger_cess`` entrena los etiquetadores de NLTK.
5. ``etiquetar_con_nltk`` etiqueta los mismos tokens con NLTK.
6. ``construir_tabla`` une reseñas + tokens + etiquetas + coincidencia.
7. ``guardar_resultados`` genera los dos CSV de salida.

Dependencias
------------
Instalar antes de ejecutar::

    pip install pandas spacy nltk
    python -m spacy download es_core_news_md

El corpus ``cess_esp`` de NLTK se descarga automáticamente.

Ejemplo de ejecución
--------------------
::

    python src/pos_tagging.py

Salidas (en ``data/results/``)
----------------------------

Cómo se comparan las etiquetas
------------------------------
Cada fila (token) tiene la etiqueta UPOS de spaCy (``pos_spacy``) y la de
NLTK (``pos_nltk``). La columna ``coincide`` es ``True`` cuando ambas son
iguales. El porcentaje de coincidencia global es el promedio de esa columna.
"""

from pathlib import Path

import nltk
import pandas as pd
import spacy
from nltk.corpus import cess_esp
from nltk.tag import BigramTagger, DefaultTagger, UnigramTagger

# Raíz del proyecto: src/pos_tagging.py -> padre de src/
BASE_DIR = Path(__file__).resolve().parents[1]
RUTA_ENTRADA = BASE_DIR / "data/processed/processed_es.csv"
RUTA_SALIDA = BASE_DIR / "data/results/pos_analysis.csv"
RUTA_SALIDA_2 = BASE_DIR / "data/results/pos_analysis_2.csv"

# Nombre exacto de la columna de texto en el archivo de entrada
COLUMNA_TEXTO = "review_text_es"
# Modelo de spaCy en español
MODELO_SPACY = "es_core_news_md"
# Columnas del análisis POS y su orden en pos_analysis_2.csv
COLUMNAS_ANALISIS = ["review_id", "token", "pos_spacy", "pos_nltk", "coincide"]


def eagles_a_upos(etiqueta):
    """Convierte una etiqueta EAGLES del corpus cess_esp a una etiqueta UPOS."""
    etiqueta = etiqueta.lower()
    # Casos especiales de varias letras que se revisan antes que el mapa
    if etiqueta.startswith("np"):
        return "PROPN"
    if etiqueta.startswith("cs"):
        return "SCONJ"
    if etiqueta.startswith(("va", "vs")):
        return "AUX"  # formas de "haber" y "ser"
    # Mapa a partir de la primera letra de la etiqueta
    mapa = {
        "a": "ADJ", "c": "CCONJ", "d": "DET", "f": "PUNCT", "i": "INTJ",
        "n": "NOUN", "p": "PRON", "r": "ADV", "s": "ADP", "v": "VERB",
        "z": "NUM", "w": "NUM",
    }
    return mapa.get(etiqueta[:1], "X")


def cargar_textos(ruta):
    """Lee el CSV de entrada y valida que exista la columna de texto."""
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: {ruta}. "
            "Verifica que exista processed_es.csv."
        )
    datos = pd.read_csv(ruta)
    if COLUMNA_TEXTO not in datos.columns:
        raise KeyError(
            f"La columna '{COLUMNA_TEXTO}' no está en el archivo {ruta}. "
            f"Columnas disponibles: {list(datos.columns)}"
        )
    return datos


def cargar_modelo_spacy(nombre):
    """Carga el modelo de spaCy en español sin el componente de entidades (ner)."""
    try:
        # Se desactiva 'ner' para acelerar el proceso; el parser se mantiene
        # activo porque se necesita para segmentar oraciones con doc.sents.
        return spacy.load(nombre, disable=["ner"])
    except OSError as error:
        raise OSError(
            f"No se pudo cargar el modelo '{nombre}'. "
            f"Instálalo con: python -m spacy download {nombre}"
        ) from error


def etiquetar_con_spacy(nlp, datos):
    """Etiqueta los textos con spaCy y devuelve una fila por token."""
    filas = []
    for review_id, doc in zip(datos.index, nlp.pipe(datos[COLUMNA_TEXTO], batch_size=64)):
        for oracion in doc.sents:
            for token in oracion:
                filas.append({
                    "review_id": review_id,
                    "token": token.text,
                    "pos_spacy": token.pos_,
                })
    return pd.DataFrame(filas, columns=["review_id", "token", "pos_spacy"])


def entrenar_tagger_cess():
    """Entrena los taggers de NLTK con cess_esp y devuelve sus exactitudes."""
    nltk.download("cess_esp", quiet=True)
    # Se convierte cada etiqueta EAGLES a UPOS y se descartan los sujetos
    # elididos que el corpus marca como "*0*".
    oraciones = [
        [(palabra.lower(), eagles_a_upos(etiq)) for palabra, etiq in oracion if palabra != "*0*"]
        for oracion in cess_esp.tagged_sents()
    ]
    corte = int(len(oraciones) * 0.9)
    entreno, prueba = oraciones[:corte], oraciones[corte:]

    unigrama = UnigramTagger(entreno, backoff=DefaultTagger("NOUN"))
    bigrama = BigramTagger(entreno, backoff=unigrama)

    exactitud_unigrama = unigrama.accuracy(prueba)
    exactitud_bigrama = bigrama.accuracy(prueba)
    return bigrama, exactitud_unigrama, exactitud_bigrama


def etiquetar_con_nltk(bigrama, tokens):
    """Etiqueta con NLTK los mismos tokens generados por spaCy."""
    # Se usan los tokens en minúsculas porque así se entrenó el corpus.
    etiquetas = [etiqueta for _, etiqueta in bigrama.tag(tokens["token"].str.lower().tolist())]
    tokens = tokens.copy()
    tokens["pos_nltk"] = etiquetas
    return tokens


def construir_tabla(datos, tokens):
    """Une las reseñas originales con los tokens y calcula la coincidencia."""
    tokens = tokens.copy()
    tokens["coincide"] = tokens["pos_spacy"] == tokens["pos_nltk"]
    # Se conservan todas las columnas originales; review_id es el índice de fila.
    resenas = datos.reset_index(names="review_id")
    return resenas.merge(tokens, on="review_id", how="inner")


def guardar_resultados(tabla, ruta):
    """Crea la carpeta de salida si no existe y guarda el DataFrame en CSV."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False, encoding="utf-8")
    return ruta.resolve()


def main():
    """Ejecuta el etiquetado POS completo y genera los archivos de salida."""
    datos = cargar_textos(RUTA_ENTRADA)
    nlp = cargar_modelo_spacy(MODELO_SPACY)
    tokens = etiquetar_con_spacy(nlp, datos)

    bigrama, exactitud_unigrama, exactitud_bigrama = entrenar_tagger_cess()
    tokens = etiquetar_con_nltk(bigrama, tokens)

    tabla = construir_tabla(datos, tokens)
    ruta_1 = guardar_resultados(tabla, RUTA_SALIDA)
    ruta_2 = guardar_resultados(tabla[COLUMNAS_ANALISIS], RUTA_SALIDA_2)

    coincidencia = tabla["coincide"].mean() * 100
    print(f"Reseñas procesadas: {len(datos)}")
    print(f"Tokens procesados: {len(tabla)}")
    print(f"Exactitud UnigramTagger: {exactitud_unigrama:.4f}")
    print(f"Exactitud BigramTagger (backoff): {exactitud_bigrama:.4f}")
    print(f"Coincidencia spaCy vs NLTK: {coincidencia:.2f}%")
    print(f"Archivo 1: {ruta_1}")
    print(f"Archivo 2: {ruta_2}")


if __name__ == "__main__":
    main()
