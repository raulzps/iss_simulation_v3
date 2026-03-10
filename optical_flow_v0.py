#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo de flujo óptico entre imágenes ISS georreferenciadas y VIIRS (pipeline v3).

Para cada imagen georreferenciada:
  1. Lee la imagen ISS *_rect.tiff (RGB) desde geo_dir.
  2. Lee la imagen VIIRS correspondiente *_viirs.tiff desde viirs_dir.
  3. Redimensiona VIIRS a la resolución de la ISS.
  4. Preprocesa: normaliza VIIRS, pasa ISS a gris y hace match de histogramas.
  5. (Opcional) Aplica un recorte definido en coordenadas normalizadas [0,1] (x,y).
  6. Calcula flujo óptico denso (Farneback) REFERENCE→DISTORTED.
  7. Guarda el flujo (u,v) en un .npy en flow_dir.

Uso típico en la pipeline:

    python -m scripts_v3.optical_flow \
        --geo_dir ISS053-E-18900-19385/geo \
        --viirs_dir ISS053-E-18900-19385/viirs_cropped_aligned \
        --flow_dir ISS053-E-18900-19385/flow \
        --start_id 19358 \
        --end_id 19385 \
        --plot_every 50 \
        --crop_x_start 0.0 --crop_x_end 1.0 \
        --crop_y_start 0.0 --crop_y_end 1.0
"""

import os
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import exposure
import flow_vis
import rasterio


def compute_and_save_optical_flow(
    image_id: str,
    geo_dir: str,
    viirs_dir: str,
    flow_dir: str,
    crop_x_start: float,
    crop_x_end: float,
    crop_y_start: float,
    crop_y_end: float,
    show_plot: bool = False,
):
    """
    Calcula y guarda el flujo óptico para un ID de imagen dado.

    image_id        : base name, p.ej. 'ISS067-E-201283'
    geo_dir         : carpeta con imágenes ISS georreferenciadas ( *_rect.tiff )
    viirs_dir       : carpeta con imágenes VIIRS alineadas ( *_viirs.tiff )
    flow_dir        : carpeta donde se guardan los .npy
    crop_*          : fracciones [0,1] para recorte (x,y). Por defecto 0,1 → sin recorte.
    show_plot       : si True, guarda un PNG de visualización del flujo.
    """
    iss2_file = os.path.join(geo_dir, f"{image_id}_rect.tiff")
    viirs_file = os.path.join(viirs_dir, f"{image_id}_viirs.tiff")
    flow_outfile = os.path.join(flow_dir, f"{image_id}_flow.npy")

    try:
        if not os.path.exists(iss2_file):
            raise FileNotFoundError(f"No se encontró la imagen ISS: {iss2_file}")
        if not os.path.exists(viirs_file):
            raise FileNotFoundError(f"No se encontró la imagen VIIRS: {viirs_file}")

        # Leer ISS (RGB)
        with rasterio.open(iss2_file) as src:
            # Asumimos 3 bandas RGB
            ISS2 = src.read([1, 2, 3])
            ISS2 = np.transpose(ISS2, (1, 2, 0))  # (H, W, C)

        # Leer VIIRS (banda única)
        with rasterio.open(viirs_file) as src:
            VIIRS = src.read(1)

        # Redimensionar VIIRS para que coincida con ISS2
        VIIRS = np.nan_to_num(VIIRS, nan=0, posinf=0, neginf=0)
        h_iss, w_iss = ISS2.shape[:2]
        VIIRS = cv2.resize(VIIRS, (w_iss, h_iss), interpolation=cv2.INTER_LINEAR)

        # Normalizar VIIRS a [0, 255]
        vmax = np.max(VIIRS)
        if vmax <= 0:
            raise ValueError(f"VIIRS para {image_id} tiene máximo <= 0, imposible normalizar.")
        VIIRS_u8 = np.uint8(VIIRS / vmax * 255)

        # Convertir ISS2 a escala de grises y emparejar histograma con VIIRS
        # OJO: rasterio lee en (R,G,B), pero cv2 espera BGR → da igual para gris, pero mantenemos forma:
        iss_gray = cv2.cvtColor(ISS2, cv2.COLOR_RGB2GRAY)
        iss_gray_matched = exposure.match_histograms(iss_gray, VIIRS_u8)

        # Seleccionar REFERENCE y DISTORTED
        REFERENCE = VIIRS_u8
        DISTORTED = iss_gray_matched

        h, w = REFERENCE.shape

        # Aplicar recorte normalizado si no es el cuadro completo
        # (0.0, 1.0, 0.0, 1.0) → sin recorte
        x0 = int(w * crop_x_start)
        x1 = int(w * crop_x_end)
        y0 = int(h * crop_y_start)
        y1 = int(h * crop_y_end)

        # Asegurar rangos válidos
        x0 = max(0, min(x0, w - 1))
        x1 = max(x0 + 1, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(y0 + 1, min(y1, h))

        REFERENCE_crop = REFERENCE[y0:y1, x0:x1]
        DISTORTED_crop = DISTORTED[y0:y1, x0:x1]

        # Calcular flujo óptico (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            REFERENCE_crop,
            DISTORTED_crop,
            None,
            0.5,   # pyr_scale
            3,     # levels
            19,    # winsize
            3,     # iterations
            7,     # poly_n
            1.5,   # poly_sigma
            0,     # flags
        )

        # Guardar flujo
        os.makedirs(flow_dir, exist_ok=True)
        np.save(flow_outfile, flow)
        print(f"✅ Flujo guardado: {flow_outfile}")

        # Visualización (opcional): guarda PNG en disco
        if show_plot:
            flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
            plt.figure(figsize=(6, 6))
            plt.imshow(flow_color)
            plt.title(f"Flujo óptico: {image_id}")
            plt.axis("off")

            plot_path = os.path.join(flow_dir, f"{image_id}_flow_vis.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"🖼  Visualización guardada: {plot_path}")

    except Exception as e:
        print(f"❌ Error con {image_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Cálculo de flujo óptico entre imágenes ISS (geo) y VIIRS (pipeline v3)."
    )
    parser.add_argument(
        "--geo_dir",
        type=str,
        required=True,
        help="Directorio con las imágenes ISS georreferenciadas (*_rect.tiff).",
    )
    parser.add_argument(
        "--viirs_dir",
        type=str,
        required=True,
        help="Directorio con las imágenes VIIRS alineadas (*_viirs.tiff).",
    )
    parser.add_argument(
        "--flow_dir",
        type=str,
        required=True,
        help="Directorio de salida para los archivos de flujo (*.npy).",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        required=True,
        help="ID inicial (última parte numérica del nombre, p.ej. 201283).",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        required=True,
        help="ID final.",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=100,
        help="Guardar una visualización PNG cada N imágenes (0 para desactivar).",
    )
    parser.add_argument(
        "--crop_x_start",
        type=float,
        default=0.0,
        help="Fracción inicial eje X (0.0 = borde izquierdo).",
    )
    parser.add_argument(
        "--crop_x_end",
        type=float,
        default=1.0,
        help="Fracción final eje X (1.0 = borde derecho).",
    )
    parser.add_argument(
        "--crop_y_start",
        type=float,
        default=0.0,
        help="Fracción inicial eje Y (0.0 = parte superior).",
    )
    parser.add_argument(
        "--crop_y_end",
        type=float,
        default=1.0,
        help="Fracción final eje Y (1.0 = parte inferior).",
    )

    args = parser.parse_args()

    geo_dir = os.path.abspath(args.geo_dir)
    viirs_dir = os.path.abspath(args.viirs_dir)
    flow_dir = os.path.abspath(args.flow_dir)

    os.makedirs(flow_dir, exist_ok=True)

    # Listar IDs a partir de los *_rect.tiff en geo_dir
    all_ids = sorted(
        f.replace("_rect.tiff", "")
        for f in os.listdir(geo_dir)
        if f.endswith("_rect.tiff")
    )

    # Filtrar por rango numérico en el ID (última parte separada por '-')
    ids = [
        img_id
        for img_id in all_ids
        if img_id.split("-")[-1].isdigit()
        and args.start_id <= int(img_id.split("-")[-1]) <= args.end_id
    ]

    print(f"🔵 Encontrados {len(ids)} IDs en rango [{args.start_id}, {args.end_id}].")

    for idx, image_id in enumerate(ids):
        show_plot = (args.plot_every > 0 and idx % args.plot_every == 0)
        compute_and_save_optical_flow(
            image_id=image_id,
            geo_dir=geo_dir,
            viirs_dir=viirs_dir,
            flow_dir=flow_dir,
            crop_x_start=args.crop_x_start,
            crop_x_end=args.crop_x_end,
            crop_y_start=args.crop_y_start,
            crop_y_end=args.crop_y_end,
            show_plot=show_plot,
        )

    print("✅ Cálculo de flujo óptico completado.")


if __name__ == "__main__":
    main()