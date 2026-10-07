"""Traducción de reseñas con MarianMT, pensada para Google Colab (GPU).

Archivos necesarios (subir con `from google.colab import files` o clonando
el repo):
    - traducir.py
    - traduccion_1.py
    - processed.csv  (salida de principal.py)

Ejemplo en Colab (runtime con GPU: Runtime > Change runtime type > GPU):

    !pip install -q transformers sentencepiece
    !python traducir.py --entrada processed.csv --salida resenas_es.csv --lote 64

Uso local:

    python src/traducir.py --entrada data/processed/processed.csv \
        --salida data/processed/resenas_es.csv

La corrida es reanudable: cada --parcial lotes se guarda el avance en
<salida>_parcial.csv; si se interrumpe, basta con volver a lanzar el mismo
comando para continuar por donde quedó (las reseñas con review_text_es ya
relleno no se repiten).
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Permite ejecutar el script aunque no se lance desde src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from traduccion_1 import armar_oraciones, cargar_traductor, traducir_lote

log = logging.getLogger("traduccion")

COLUMNA = "review_text_es"
REINTENTOS = 3
ESPERA = 5


def argumentos():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrada", default="processed.csv",
                        help="CSV con las reseñas a traducir (por defecto: processed.csv)")
    parser.add_argument("--salida", default="resenas_es.csv",
                        help="CSV de salida con la columna review_text_es")
    parser.add_argument("--modelo", default="Helsinki-NLP/opus-mt-en-es",
                        help="Modelo de Hugging Face a utilizar")
    parser.add_argument("--lote", type=int, default=32,
                        help="Oraciones por lote (subirlo en GPU, p. ej. 64)")
    parser.add_argument("--beams", type=int, default=4,
                        help="Beam search: 4 = mejor calidad, 1 = más rápido")
    parser.add_argument("--parcial", type=int, default=100,
                        help="Cada cuántos lotes se guarda el avance (0 = sin parcial)")
    return parser.parse_args()


def ruta_parcial(salida):
    salida = Path(salida)
    return salida.with_name(salida.stem + "_parcial.csv" + "".join(salida.suffixes[1:]))


def cargar_avance(entrada, salida):
    """Carga las reseñas y recupera el avance de una corrida anterior."""
    df = pd.read_csv(entrada)
    if COLUMNA not in df.columns:
        df[COLUMNA] = ""

    for fuente in (Path(salida), ruta_parcial(salida)):
        if not fuente.exists():
            continue
        previo = pd.read_csv(fuente)
        if len(previo) == len(df) and COLUMNA in previo.columns:
            df[COLUMNA] = previo[COLUMNA].fillna("").values
            log.info("Avance recuperado de %s (%d/%d reseñas ya traducidas)",
                     fuente, int((df[COLUMNA].astype(str).str.strip() != "").sum()), len(df))
            return df
        log.warning("Se ignora %s: no coincide con el CSV de entrada", fuente)
    return df


def preparar_pendientes(df):
    """Devuelve las reseñas sin traducir y sus oraciones."""
    texto = df[COLUMNA]
    pendientes = df[texto.isna() | (texto.astype(str).str.strip() == "")].copy()
    oraciones = armar_oraciones(pendientes)

    # Reseñas sin oraciones no se pueden traducir: se marcan con su texto original
    con_oraciones = {indice for indice, _, _ in oraciones}
    sin_oraciones = [i for i in pendientes.index if i not in con_oraciones]
    if sin_oraciones:
        df.loc[sin_oraciones, COLUMNA] = (
            df.loc[sin_oraciones, "review_text"].fillna("").astype(str))
        pendientes = pendientes.drop(sin_oraciones)

    return pendientes, oraciones


def unir_resultados(df, oraciones, resultados, parcial):
    """Combina las oraciones traducidas con el resto de las reseñas.

    En un parcial, las oraciones aún no traducidas quedan vacías para que la
    reanudación las vuelva a procesar; en la salida final se dejan en inglés.
    """
    por_resena = {}
    for (indice, posicion, original), traducida in zip(oraciones, resultados):
        if traducida is not None:
            valor = traducida
        else:
            valor = "" if parcial else original
        por_resena.setdefault(indice, []).append((posicion, valor))

    df = df.copy()
    actuales = df[COLUMNA]
    df[COLUMNA] = [
        " ".join(t for _, t in sorted(por_resena[i])) if i in por_resena
        else (e if isinstance(e, str) else "")
        for i, e in zip(df.index, actuales)
    ]
    return df


def traducir(df, pendientes, oraciones, args, salida_parcial):
    tokenizador, modelo, dispositivo = cargar_traductor(args.modelo)

    # Se ordena por longitud para que cada lote sea más eficiente
    orden = sorted(range(len(oraciones)), key=lambda i: len(oraciones[i][2]))
    resultados = [None] * len(orden)

    for inicio in range(0, len(orden), args.lote):
        posiciones = orden[inicio:inicio + args.lote]
        textos = [oraciones[i][2] for i in posiciones]
        traducidos = traducir_lote(textos, tokenizador, modelo, dispositivo,
                                   REINTENTOS, ESPERA, num_beams=args.beams)
        for i, texto in zip(posiciones, traducidos):
            resultados[i] = texto

        lote_num = inicio // args.lote
        if lote_num % 10 == 0:
            log.info("Progreso: %d/%d oraciones",
                     min(inicio + args.lote, len(orden)), len(orden))
        if args.parcial and lote_num and lote_num % args.parcial == 0:
            parcial = unir_resultados(df, oraciones, resultados, parcial=True)
            salida_parcial.parent.mkdir(parents=True, exist_ok=True)
            parcial.to_csv(salida_parcial, index=False, encoding="utf-8")
            log.info("Avance guardado: %s", salida_parcial)

    return resultados


def main():
    args = argumentos()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    df = cargar_avance(args.entrada, args.salida)
    pendientes, oraciones = preparar_pendientes(df)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    if not oraciones:
        df.to_csv(salida, index=False, encoding="utf-8")
        log.info("Nada pendiente: %s ya está completa", salida)
        return

    log.info("Oraciones a traducir: %d (de %d reseñas)", len(oraciones), len(pendientes))
    resultados = traducir(df, pendientes, oraciones, args, ruta_parcial(args.salida))

    salida_df = unir_resultados(df, oraciones, resultados, parcial=False)
    salida_df.to_csv(salida, index=False, encoding="utf-8")
    fallidas = sum(r is None for r in resultados)
    log.info("Traducción terminada: %s (oraciones sin traducir: %d)", salida, fallidas)


if __name__ == "__main__":
    main()
