#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrección de puntos usando flujo óptico (pipeline v3).

Toma:
  - Archivos .points (ya filtrados y renombrados), típicamente de:
        scripts_v3.filter_rename
  - Archivos de flujo óptico *_flow.npy, generados por:
        scripts_v3.optical_flow
  - Archivos *_pixel_mapping.csv y *_rect.tiff, generados por:
        scripts_v3.georef_timelapse

Para cada imagen:
  1. Asocia cada punto del .points con su posición en píxeles (geoX, geoY)
     en la imagen georreferenciada (desde *_pixel_mapping.csv).
  2. Toma el flujo óptico en esa posición (corrigiendo si hubo recorte en optical_flow).
  3. Desplaza la posición (en píxeles) según el flujo.
  4. Convierte las nuevas posiciones de píxeles a coordenadas geográficas
     (mapX, mapY) vía la geotransformación GDAL de *_rect.tiff.
  5. Guarda un nuevo archivo *_corrected.points con los campos:
        mapX, mapY, sourceX, sourceY, enable, dX, dY, residual

Usar SIEMPRE los mismos parámetros de recorte (crop_x_start, etc.)
que se usaron en scripts_v3.optical_flow.

Ejemplo de uso:

    python -m scripts_v3.correct_points \
        --input_points_dir /ruta/base/filtered_points \
        --flow_dir /ruta/base/flow \
        --geo_dir /ruta/base/geo \
        --output_dir /ruta/base/corrected_points \
        --start_id 201283 \
        --end_id 202059 \
        --crop_x_start 0.0 --crop_x_end 1.0 \
        --crop_y_start 0.0 --crop_y_end 1.0
"""

import os
import argparse

import numpy as np
import pandas as pd
from osgeo import gdal


def extract_id_from_filename(filename: str):
    """
    Extrae el ID numérico final de nombres tipo 'ISS067-E-201283.points'.

    Devuelve int o None si no se puede.
    """
    try:
        base = os.path.splitext(os.path.basename(filename))[0]
        # base ≈ 'ISS067-E-201283' → nos quedamos con lo posterior al último '-'
        return int(base.split("-")[-1])
    except Exception:
        return None


def pixel_to_geo(px: float, py: float, geotransform):
    """
    Convierte coordenadas de píxel (col=px, fila=py) a coordenadas geográficas (mapX, mapY)
    usando la geotransformación GDAL.
    """
    mapX = geotransform[0] + px * geotransform[1] + py * geotransform[2]
    mapY = geotransform[3] + px * geotransform[4] + py * geotransform[5]
    return mapX, mapY


def correct_points_with_flow(
    points_path: str,
    flow_path: str,
    mapping_file: str,
    geo_image_path: str,
    corrected_points_path: str,
    crop_x_start: float,
    crop_x_end: float,
    crop_y_start: float,
    crop_y_end: float,
):
    """
    Corrige un único archivo .points usando el flujo óptico y el mapeo píxel↔geo.

    points_path          : archivo .points de entrada (sin corregir)
    flow_path            : archivo .npy con el flujo (u,v) en el recorte escogido
    mapping_file         : CSV con columnas ['sourceX','sourceY','mapX','mapY','geoX','geoY']
                           generado por georef_timelapse.
    geo_image_path       : tiff georreferenciado *_rect.tiff
    corrected_points_path: ruta del .points corregido a escribir
    crop_*               : mismos parámetros de recorte usados en optical_flow.py
    """
    # Leer puntos originales
    points = pd.read_csv(points_path)

    # Leer mapping (pixel_mapping.csv): asocia cada GCP con (geoX, geoY) en la imagen georreferenciada
    mapping = pd.read_csv(mapping_file)

    if 'geoX' not in mapping.columns or 'geoY' not in mapping.columns:
        raise ValueError(
            f"El archivo de mapeo no contiene columnas 'geoX' y 'geoY': {mapping_file}"
        )

    if len(mapping) != len(points):
        print(
            f"⚠️ Aviso: mapping ({len(mapping)}) y points ({len(points)}) "
            f"tienen distinta longitud. Se asumirá alineación por índice."
        )

    # Añadimos columnas de posición de píxel en la imagen georreferenciada
    points['geoX'] = mapping['geoX']
    points['geoY'] = mapping['geoY']

    # Cargamos el flujo óptico
    flow = np.load(flow_path)  # shape (Hc, Wc, 2) sobre el recorte

    # Abrimos la imagen georreferenciada para obtener geotransform y tamaño
    dataset = gdal.Open(geo_image_path)
    if dataset is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen georreferenciada: {geo_image_path}")
    geotransform = dataset.GetGeoTransform()
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # Recalcular el recorte en píxeles, igual que en optical_flow.py
    x0 = int(width * crop_x_start)
    x1 = int(width * crop_x_end)
    y0 = int(height * crop_y_start)
    y1 = int(height * crop_y_end)

    # Normalizar a rangos válidos
    x0 = max(0, min(x0, width - 1))
    x1 = max(x0 + 1, min(x1, width))
    y0 = max(0, min(y0, height - 1))
    y1 = max(y0 + 1, min(y1, height))

    crop_w = x1 - x0
    crop_h = y1 - y0

    # Comprobación de coherencia con el flujo
    if flow.shape[1] != crop_w or flow.shape[0] != crop_h:
        print(
            f"⚠️ Atención: dimensiones del flujo {flow.shape[1]}x{flow.shape[0]} "
            f"no coinciden con el recorte esperado {crop_w}x{crop_h}. "
            f"Revisa que los parámetros crop_* coincidan con optical_flow.py."
        )

    corrected_mapX = []
    corrected_mapY = []
    dX = []
    dY = []

    for geoX_pix, geoY_pix in zip(points['geoX'], points['geoY']):
        # Posición entera de píxel en la imagen georreferenciada
        x_int = int(round(geoX_pix))
        y_int = int(round(geoY_pix))

        # Coordenadas dentro del recorte donde se calculó el flujo (relativas a (x0,y0))
        rel_x = x_int - x0
        rel_y = y_int - y0

        if 0 <= rel_x < flow.shape[1] and 0 <= rel_y < flow.shape[0]:
            # flow[rel_y, rel_x] = (u,v) desplazamiento (x,y). Usamos signo negativo como en tu código original.
            dx_pix, dy_pix = -flow[rel_y, rel_x]
            new_geoX_pix = geoX_pix + dx_pix
            new_geoY_pix = geoY_pix + dy_pix
        else:
            # Fuera del recorte: no corregimos
            dx_pix, dy_pix = 0.0, 0.0
            new_geoX_pix, new_geoY_pix = geoX_pix, geoY_pix

        # Convertir a coordenadas geográficas con la geotransformación
        new_mapX, new_mapY = pixel_to_geo(new_geoX_pix, new_geoY_pix, geotransform)
        corrected_mapX.append(new_mapX)
        corrected_mapY.append(new_mapY)
        dX.append(dx_pix)
        dY.append(dy_pix)

    # Actualizar puntos
    points['mapX'] = corrected_mapX
    points['mapY'] = corrected_mapY
    points['dX'] = dX
    points['dY'] = dY
    points['residual'] = 0.0
    points['enable'] = 1

    # Reordenar columnas al formato QGIS que usas
    points = points[['mapX', 'mapY', 'sourceX', 'sourceY', 'enable', 'dX', 'dY', 'residual']]

    # Guardar
    os.makedirs(os.path.dirname(corrected_points_path), exist_ok=True)
    points.to_csv(corrected_points_path, index=False)
    print(f"✅ Corregido: {corrected_points_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Corrección de puntos con flujo óptico (pipeline v3)."
    )
    parser.add_argument(
        "--input_points_dir",
        type=str,
        required=True,
        help="Directorio con los archivos .points de entrada (filtrados/renombrados).",
    )
    parser.add_argument(
        "--flow_dir",
        type=str,
        required=True,
        help="Directorio con archivos de flujo *_flow.npy.",
    )
    parser.add_argument(
        "--geo_dir",
        type=str,
        required=True,
        help="Directorio con imágenes georreferenciadas *_rect.tiff y *_pixel_mapping.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directorio de salida para *_corrected.points.",
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
        "--crop_x_start",
        type=float,
        default=0.0,
        help="Fracción inicial eje X usada en optical_flow (0.0 = borde izquierdo).",
    )
    parser.add_argument(
        "--crop_x_end",
        type=float,
        default=1.0,
        help="Fracción final eje X usada en optical_flow (1.0 = borde derecho).",
    )
    parser.add_argument(
        "--crop_y_start",
        type=float,
        default=0.0,
        help="Fracción inicial eje Y usada en optical_flow (0.0 = parte superior).",
    )
    parser.add_argument(
        "--crop_y_end",
        type=float,
        default=1.0,
        help="Fracción final eje Y usada en optical_flow (1.0 = parte inferior).",
    )

    args = parser.parse_args()

    input_points_dir = os.path.abspath(args.input_points_dir)
    flow_dir = os.path.abspath(args.flow_dir)
    geo_dir = os.path.abspath(args.geo_dir)
    output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Listar todos los .points y filtrar por rango de IDs
    all_point_files = sorted(
        f for f in os.listdir(input_points_dir) if f.endswith(".points")
    )

    selected_files = []
    for fname in all_point_files:
        file_id = extract_id_from_filename(fname)
        if file_id is not None and args.start_id <= file_id <= args.end_id:
            selected_files.append(fname)

    print(f"🔵 Archivos .points a procesar en rango [{args.start_id}, {args.end_id}]: {len(selected_files)}")

    for fname in selected_files:
        base_name = os.path.splitext(fname)[0]  # p.ej. 'ISS067-E-201283'

        points_path = os.path.join(input_points_dir, fname)
        flow_path = os.path.join(flow_dir, f"{base_name}_flow.npy")
        mapping_file = os.path.join(geo_dir, f"{base_name}_pixel_mapping.csv")
        geo_image_path = os.path.join(geo_dir, f"{base_name}_rect.tiff")
        corrected_points_path = os.path.join(output_dir, f"{base_name}_corrected.points")

        if not os.path.exists(flow_path):
            print(f"⚠️ Sin flujo óptico para {base_name}, se omite: {flow_path}")
            continue
        if not os.path.exists(mapping_file):
            print(f"⚠️ Sin pixel_mapping.csv para {base_name}, se omite: {mapping_file}")
            continue
        if not os.path.exists(geo_image_path):
            print(f"⚠️ Sin imagen georreferenciada para {base_name}, se omite: {geo_image_path}")
            continue

        try:
            correct_points_with_flow(
                points_path=points_path,
                flow_path=flow_path,
                mapping_file=mapping_file,
                geo_image_path=geo_image_path,
                corrected_points_path=corrected_points_path,
                crop_x_start=args.crop_x_start,
                crop_x_end=args.crop_x_end,
                crop_y_start=args.crop_y_start,
                crop_y_end=args.crop_y_end,
            )
        except Exception as e:
            print(f"❌ Error corrigiendo {base_name}: {e}")

    print("✅ Corrección de puntos completada.")


if __name__ == "__main__":
    main()