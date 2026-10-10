"""Traduce una columna de `processed.csv` al espanol y guarda el resultado.

Uso:
    python traducir.py                      # processed.csv -> resenas_es.csv (todo)
    python traducir.py --limite 20          # prueba rapida con las 20 primeras
    python traducir.py --beam 4             # beam search (mas lento, mejor calidad)
    python traducir.py --columna review_summary

Es reanudable: guarda cada `--bloque` filas, y si el CSV de salida ya existe
se salta las filas ya traducidas.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from traductor import cargar_modelo, traducir_resenas

COLUMNA_DESTINO = "_es"


def argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entrada", default="processed.csv")
    p.add_argument("--salida", default=None, help="CSV de salida (por defecto: <entrada>_es.csv)")
    p.add_argument("--columna", default="review_text", help="columna a traducir")
    p.add_argument("--limite", type=int, default=None, help="traducir solo las N primeras filas")
    p.add_argument("--lote", type=int, default=64, help="oraciones por tanda de traduccion")
    p.add_argument("--beam", type=int, default=1, help="num_beams (1=greedy rapido, 4=mejor calidad)")
    p.add_argument("--bloque", type=int, default=500, help="filas por guardado parcial")
    p.add_argument("--sin-cuantizar", action="store_true", help="no usar int8 en CPU (mas lento, ~6x)")
    p.add_argument("--sin-reanudar", action="store_true", help="ignorar lo ya traducido en la salida")
    return p.parse_args()


def main():
    args = argumentos()
    entrada = Path(args.entrada)
    salida = Path(args.salida) if args.salida else entrada.with_name(entrada.stem + COLUMNA_DESTINO + entrada.suffix)
    destino = args.columna + COLUMNA_DESTINO

    if not entrada.exists():
        raise SystemExit(f"No existe el archivo de entrada: {entrada}")

    df = pd.read_csv(entrada)
    if args.columna not in df.columns:
        raise SystemExit(f"La columna '{args.columna}' no existe. Disponibles: {list(df.columns)}")
    df[args.columna] = df[args.columna].fillna("").astype(str)
    if args.limite:
        df = df.head(args.limite).copy()

    # Reanudacion: si la salida ya existe, reutilizamos lo ya traducido.
    hechos = 0
    if salida.exists() and not args.sin_reanudar:
        previo = pd.read_csv(salida)
        if destino in previo.columns:
            hechos = min(int(previo[destino].notna().sum()), len(df))
            if hechos:
                print(f"Reanudando: {hechos} filas ya traducidas en {salida}")
                df[destino] = pd.Series([None] * len(df), index=df.index, dtype=object)
                df.loc[df.index[:hechos], destino] = previo[destino].iloc[:hechos].to_numpy()

    pendientes = len(df) - hechos
    if pendientes <= 0:
        print(f"Nada que traducir: {len(df)} filas ya estan en {salida}")
        return

    inicio = time.time()
    stats: dict = {}
    # Fijamos la cuantizacion antes de traducir: el modelo se carga una sola vez.
    cargar_modelo(cuantizar=not args.sin_cuantizar)
    print(f"Traduciendo {pendientes} filas de '{args.columna}' (lote={args.lote}, beam={args.beam})...")

    for desde in range(hechos, len(df), args.bloque):
        hasta = min(desde + args.bloque, len(df))
        indices = df.index[desde:hasta]
        df.loc[indices, destino] = traducir_resenas(
            df.loc[indices, args.columna].tolist(),
            lote=args.lote,
            beam=args.beam,
            verbose=True,
            stats=stats,
        )
        salida.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(salida, index=False)
        hechas = hasta - hechos
        elapsed = time.time() - inicio
        print(
            f"[{hasta}/{len(df)}] {hechas} filas en {elapsed:.0f}s "
            f"({elapsed / hechas:.2f}s/fila, ~{elapsed / hechas * pendientes / 60:.0f} min restantes)"
            f" -> {salida}"
        )

    print(f"Listo: {salida}  ({len(df)} filas, columna '{destino}')")
    if stats.get("recortadas"):
        print(f"Aviso: {stats['recortadas']} de {stats['oraciones']} oraciones llegaron al limite de "
              f"256 tokens y se truncaron. Sube MAX_TOKENS en traductor.py si es demasiado.")


if __name__ == "__main__":
    main()
