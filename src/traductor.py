"""Traduccion ingles -> espanol con Helsinki-NLP/opus-mt-en-es.

Modulo compartido por el script `traducir.py` y por la notebook `traducir.ipynb`.
El modelo se carga de forma perezosa: importar el modulo no descarga nada.
"""

from __future__ import annotations

import copy
import re
import warnings
from functools import lru_cache

import torch
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer

MODELO = "Helsinki-NLP/opus-mt-en-es"
LOTE = 64          # oraciones por tanda
MAX_TOKENS = 384   # tokens por oración (la ventana del modelo es de 512)


@lru_cache(maxsize=4)
def cargar_modelo(modelo: str = MODELO, cuantizar: bool | None = None):
    """Devuelve (tokenizador, modelo, device). Se descarga solo en la 1a llamada.

    En CPU cuantizamos las capas Lineales a int8: ~6x mas rapido con una perdida
    de calidad despreciable para este modelo (medido: 0.9 -> 5.2 oraciones/s).
    cuantizar=None lo decide solo segun el dispositivo; True/False lo fuerzan.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(modelo)
    mt = MarianMTModel.from_pretrained(modelo).to(device).eval()
    if device == "cpu" and (cuantizar is None or cuantizar):
        # cuantize_dynamic MUTA el modelo en sitio: cuantizamos una copia.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mt = torch.quantization.quantize_dynamic(
                copy.deepcopy(mt), {torch.nn.Linear}, dtype=torch.qint8
            ).eval()
    return tok, mt, device


def traducir_oraciones(oraciones, lote=LOTE, beam=1, verbose=False, stats=None):
    """Traduce una lista PLANA de oraciones y devuelve otra lista.

    beam=1 (greedy) es ~4x mas rapido que el beam search por defecto (4) y a esta
    escala la diferencia de calidad es pequena; usa beam=4 para la maxima calidad.
    """
    if not oraciones:
        return []

    tok, mt, device = cargar_modelo()

    # Agrupar por longitud: generate() no para hasta que TODAS las frases del lote
    # terminan, asi que mezclar cortas y largas hace que cada tanda dure lo que
    # dure la mas larga. Ordenamos por tokens y luego deshacemos el orden.
    largos = [len(ids) for ids in tok(oraciones, add_special_tokens=False)["input_ids"]]
    vacias = {i for i, largo in enumerate(largos) if largo == 0}
    orden = sorted((i for i in range(len(oraciones)) if i not in vacias), key=lambda i: largos[i])

    salida = [""] * len(oraciones)
    recorrer = _barra(len(orden), verbose)
    with recorrer as barra:
        for i in range(0, len(orden), lote):
            idx = orden[i:i + lote]
            tanda = [oraciones[j].strip() for j in idx]
            entradas = tok(
                tanda,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_length=True,
            )
            # `length` solo sirve para medir el truncado: hay que sacarlo
            # antes de generate() y antes de moverlo a la GPU.
            longitudes = entradas.pop("length").tolist()
            entradas = entradas.to(device)

            with torch.inference_mode():
                generado = mt.generate(**entradas, num_beams=beam)
            for j, texto in zip(idx, tok.batch_decode(generado, skip_special_tokens=True)):
                salida[j] = texto

            if stats is not None:
                stats["oraciones"] = stats.get("oraciones", 0) + len(tanda)
                stats["recortadas"] = stats.get("recortadas", 0) + sum(
                    1 for largo in longitudes if largo >= MAX_TOKENS
                )
            if barra is not None:
                barra.update(len(tanda))
    return salida


_SIN_ESPACIO = re.compile(r"([.!?,;:])(?=[A-Za-z])")
# inicial, sigla, "&" o numero justo antes del signo: no partir (U.S.A, Sr., n.º 3)
_ABREVIATURA = re.compile(r"(?:\b[A-Z]|&\w|\d)$")


def normalizar_espacios(texto: str) -> str:
    """Inserta el espacio que falta tras . ! ? , ; : .

    El 57% de las resenas de processed.csv viene pegada ("actors.Love this")
    y sent_tokenize NO las separa, asi que el modelo recibia parrafos pegados y
    devolvia texto sin espaciado ("los actores.Un amor").
    """
    trozos, pos = [], 0
    for m in _SIN_ESPACIO.finditer(texto):
        # Solo miramos una ventana corta antes del signo: así es O(n), no O(n²).
        if _ABREVIATURA.search(texto[max(0, m.start() - 5):m.start()].rstrip()):
            continue
        trozos.append(texto[pos:m.start()])
        trozos.append(m.group(1) + " ")
        pos = m.end()
    trozos.append(texto[pos:])
    return re.sub(r"[ \t]{2,}", " ", "".join(trozos))


def segmentar(textos):
    """Divide cada texto en oraciones. Devuelve (lista_de_listas, lista_plana)."""
    grupos = [sent_tokenize(normalizar_espacios(str(t))) for t in textos]
    return grupos, [o for grupo in grupos for o in grupo]


def traducir_resenas(textos, lote=LOTE, beam=1, verbose=False, stats=None):
    """Traduce varias resenas leveraging UN SOLO conjunto de lotes.

    Es la version eficiente: aplana todas las oraciones, las traduce en tandas
    grandes y despues reagrupa. Traducir resena por resena genera una llamada a
    generate() por oración, que es lo que hace lento al pipeline original.
    """
    grupos, plano = segmentar(textos)
    if stats is not None:
        stats["resenas"] = len(textos)
    traducidas = traducir_oraciones(plano, lote=lote, beam=beam, verbose=verbose, stats=stats)

    salida, pos = [], 0
    for grupo in grupos:
        salida.append(" ".join(traducidas[pos:pos + len(grupo)]))
        pos += len(grupo)
    return salida


def traducir_resena(texto, **kwargs):
    """Traduce una sola resena (devuelve un string)."""
    return traducir_resenas([texto], **kwargs)[0]


def _barra(total, verbose):
    if not verbose:
        return _NULO
    try:
        from tqdm import tqdm
    except ImportError:
        return _NULO
    return tqdm(total=total, unit="oracion", desc="Traduciendo")


class _NULO:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
