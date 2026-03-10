#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recorte y alineado de VIIRS a las imágenes ISS georreferenciadas (pipeline v3).

Este script:
  1) Lee las imágenes georreferenciadas de la ISS (geo_dir, típicamente *_rect.tiff).
  2) Para cada una, calcula la región de interés (ROI) válida.
  3) Recorta el mosaico VIIRS a esa ROI y normaliza valores.
  4) Reproyecta/alinea el recorte VIIRS al CRS de la imagen ISS.
  5) Guarda el resultado como <base_name>_viirs.tiff en output_dir.

Pensado para usarse desde la pipeline, por ejemplo:

    python -m scripts_v3.viirs_roi_crop \
        --geo_dir /ruta/base/geo \
        --output_dir /ruta/base/viirs_cropped_aligned \
        --viirs_tiff /ruta/al/mosaico_viirs.tif \
        --start_id 201283 \
        --end_id 202059 \
        --mode fast \
        --nproc 8

Donde 'mode' puede ser:
  - fast:  clip por bounding box + alineado mínimo, resampling nearest (rápido).
  - safe:  recorte poligonal + alineado exacto a la rejilla de la ISS, resampling bilinear (más preciso, más lento).
"""

import os
import time
import argparse
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform_geom
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tempfile
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# =========================
# Utilidades geométricas
# =========================

def compute_valid_bbox_geom(reference_raster):
    """
    Versión ultrarrápida: calcula la bbox de píxeles válidos (≠ nodata o >0 si nodata None)
    usando índices de numpy, sin construir polígonos complejos.

    Devuelve una geometría tipo GeoJSON (mapping(box(...))) en el CRS del raster de referencia.
    """
    with rasterio.open(reference_raster) as src:
        arr = src.read(1)

        if src.nodata is not None:
            valid = arr != src.nodata
        else:
            valid = arr > 0

        if not valid.any():
            raise ValueError("No hay píxeles válidos en la referencia.")

        rows, cols = np.where(valid)
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()

        win = Window.from_slices((rmin, rmax + 1), (cmin, cmax + 1))
        bminx, bminy, bmaxx, bmaxy = window_bounds(win, src.transform)

        return mapping(box(bminx, bminy, bmaxx, bmaxy))


def extract_roi_polygon(reference_raster, robust=False):
    """
    Versión polígono: extrae ROI real (zonas válidas) como geometría.

    robust=False evita make_valid() para máxima velocidad.
    robust=True intenta ser más cuidadoso con geometrías inválidas (más lento).
    """
    with rasterio.open(reference_raster) as src:
        image = src.read(1)
        mask_array = (image != src.nodata) if (src.nodata is not None) else (image > 0)

        geoms = []
        if robust:
            # Más robusto pero más lento: evita en lo posible geometrías inválidas
            from shapely.errors import TopologicalError
            for geom, val in shapes(image, mask=mask_array, transform=src.transform):
                if val != 0:
                    try:
                        geoms.append(shape(geom))
                    except TopologicalError:
                        continue
        else:
            # Rápido: confiar en shapes() y descartar zeros
            geoms = [
                shape(g)
                for g, v in shapes(image, mask=mask_array, transform=src.transform)
                if v != 0
            ]

        if not geoms:
            raise ValueError("No se pudo extraer ROI: referencia vacía.")

        merged_polygon = unary_union(geoms)
        return mapping(merged_polygon)


# =========================
# Recorte y normalización VIIRS
# =========================

def clip_and_normalize_viirs(input_viirs,
                             reference_raster,
                             clip_mode="bbox",
                             gamma=2.5,
                             min_val=0,
                             max_val=200):
    """
    Recorta VIIRS al ROI del reference y normaliza.

    clip_mode:
      - 'bbox'    → ROI = bounding box de píxeles válidos (muy rápido)
      - 'polygon' → ROI = polígono real de píxeles válidos (más preciso, más lento)

    Devuelve:
      - out_image: array 2D float32 (valores normalizados y transformados con gamma)
      - out_meta : metadatos rasterio para el recorte
      - viirs_crs: CRS original del VIIRS
    """
    # ROI en CRS del reference
    roi_geom_ref = (
        compute_valid_bbox_geom(reference_raster)
        if clip_mode == "bbox" else
        extract_roi_polygon(reference_raster, robust=False)
    )

    with rasterio.open(reference_raster) as ref_src:
        ref_crs = ref_src.crs

    with rasterio.open(input_viirs) as src:
        viirs_crs = src.crs

        # Transformar ROI al CRS del VIIRS si hiciera falta
        roi_for_viirs = roi_geom_ref
        if (viirs_crs is not None) and (ref_crs is not None) and (viirs_crs != ref_crs):
            roi_for_viirs = transform_geom(ref_crs, viirs_crs, roi_geom_ref)

        try:
            out_image, out_transform = mask(src, [roi_for_viirs], crop=True)
        except ValueError as e:
            raise RuntimeError(f"El ROI no solapa con el VIIRS ({e}).")

        # out_image tiene shape (bands, H, W). Usamos solo banda 1.
        out_image = out_image[0].astype(np.float32)

        # Recorte de valores
        out_image = np.clip(out_image, min_val, max_val)
        denom = (max_val - min_val)
        if denom <= 0:
            raise ValueError("Parámetros min_val/max_val inválidos.")

        # Normalizar [0,1] y aplicar gamma
        out_image = (out_image - min_val) / (denom + 1e-9)
        out_image = np.power(out_image, 1.0 / gamma)
        # Re-escalar (opcional) a 0–100
        out_image = np.clip(out_image * 100.0, 0, 100)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[0],
            "width": out_image.shape[1],
            "transform": out_transform,
            "dtype": "float32",
            "count": 1,
            "compress": "lzw",
            "nodata": 0.0,
        })

        return out_image, out_meta, viirs_crs


# =========================
# Alineado / reproyección
# =========================

def align_viirs(reference_path,
                viirs_image_array,
                viirs_meta,
                viirs_crs,
                mode="minimal",
                resampling=Resampling.nearest,
                gdal_threads=None,
                output_aligned_path=None):
    """
    Alinea/reproyecta el recorte VIIRS al CRS de la imagen de referencia.

    mode:
      - 'exact'   → misma rejilla (transform, width, height) que el raster de referencia.
      - 'minimal' → tamaño y rejilla mínimos necesarios para el recorte (más rápido, no garantiza
                    coincidencia píxel a píxel con el reference).

    resampling: Resampling.* de rasterio (nearest, bilinear, cubic, ...)

    gdal_threads:
      - int  → número de hilos
      - None → dejar que GDAL decida (puede ser 0 = auto)
    """
    assert output_aligned_path is not None

    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width, ref_height = ref.width, ref.height

    # Escribimos el recorte en un temporal (en su CRS/transform actual)
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmpfile:
        with rasterio.open(tmpfile.name, "w", **viirs_meta) as tmp_ds:
            tmp_ds.write(viirs_image_array[np.newaxis, :, :])

        with rasterio.open(tmpfile.name) as tmp_ds:
            # Configuración de destino según modo
            if mode == "exact":
                dst_crs = ref_crs
                dst_transform = ref_transform
                dst_width = ref_width
                dst_height = ref_height
            else:
                # 'minimal': calcular transform/size mínimo compatible con el recorte y el CRS de referencia
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    tmp_ds.crs, ref_crs, tmp_ds.width, tmp_ds.height, *tmp_ds.bounds
                )
                dst_crs = ref_crs

            profile = tmp_ds.profile.copy()
            profile.update({
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "dtype": "float32",
                "count": 1,
                "compress": "lzw",
                "nodata": 0.0,
            })

            # Control de hilos de GDAL para acelerar reproyección
            env_kwargs = {}
            if gdal_threads is not None:
                env_kwargs["GDAL_NUM_THREADS"] = str(gdal_threads)

            with rasterio.Env(**env_kwargs):
                with rasterio.open(output_aligned_path, "w", **profile) as dst:
                    reproject(
                        source=rasterio.band(tmp_ds, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=tmp_ds.transform,
                        src_crs=tmp_ds.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling,
                        num_threads=gdal_threads if gdal_threads else 0,  # 0 => auto (depende de GDAL)
                    )


# =========================
# Procesado por imagen
# =========================

def process_one_image(args_tuple):
    """
    Procesa una imagen de referencia (una ISS georreferenciada):

      - recorte+normalizado VIIRS al ROI (bbox/polygon)
      - reproyección en modo 'minimal' (rápido) o 'exact' (alineado a rejilla ref)

    args_tuple contiene:
      (ref_file, geo_dir, output_dir, viirs_tiff, clip_mode, align_mode, resampling, gdal_threads)
    """
    (
        ref_file,
        geo_dir,
        output_dir,
        viirs_tiff,
        clip_mode,
        align_mode,
        resampling,
        gdal_threads,
    ) = args_tuple

    start = time.time()
    ref_path = os.path.join(geo_dir, ref_file)
    base_name = ref_file.replace("_rect.tiff", "")
    output_aligned_path = os.path.join(output_dir, f"{base_name}_viirs.tiff")

    if os.path.exists(output_aligned_path):
        return f"🟡 {ref_file} ya existe, se omite."

    try:
        viirs_image, viirs_meta, viirs_crs = clip_and_normalize_viirs(
            viirs_tiff,
            ref_path,
            clip_mode=clip_mode,
            gamma=2.5,
            min_val=0,
            max_val=200,
        )

        align_viirs(
            ref_path,
            viirs_image,
            viirs_meta,
            viirs_crs,
            mode=align_mode,
            resampling=resampling,
            gdal_threads=gdal_threads,
            output_aligned_path=output_aligned_path,
        )

        elapsed = time.time() - start
        return f"✅ {ref_file} procesado en {elapsed:.2f} s"

    except RuntimeError as e:
        return f"⚪ {ref_file} sin solape: {e}"
    except Exception as e:
        return f"❌ Error en {ref_file}: {e}"


def process_timelapse_parallel(geo_dir,
                               output_dir,
                               viirs_tiff,
                               start_id,
                               end_id,
                               nproc,
                               clip_mode,
                               align_mode,
                               resampling,
                               gdal_threads):
    """
    Lanza procesamiento en paralelo.

    Filtra archivos en geo_dir por:
      - nombre que termina en '_rect.tiff'
      - ID numérico extraído del nombre (parte final antes de '_rect.tiff')
      - ID dentro de [start_id, end_id]
    """

    def extract_id_from_filename(filename: str):
        try:
            # Espera algo tipo 'ISS053-E-201283_rect.tiff'
            return int(filename.split("-")[-1].split("_")[0])
        except Exception:
            return None

    reference_files = sorted(
        f
        for f in os.listdir(geo_dir)
        if f.endswith("_rect.tiff")
        and (file_id := extract_id_from_filename(f)) is not None
        and start_id <= file_id <= end_id
    )

    print(f"🔵 Se encontraron {len(reference_files)} imágenes para procesar.")
    max_processes = min(nproc, cpu_count())
    print(f"🔵 Usando {max_processes} procesos. (GDAL threads: {gdal_threads})\n")

    # Convertir resampling_cli string a enumeración rasterio
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    resampling_enum = resampling_map[resampling]

    job_args = [
        (
            f,
            geo_dir,
            output_dir,
            viirs_tiff,
            clip_mode,
            align_mode,
            resampling_enum,
            gdal_threads,
        )
        for f in reference_files
    ]

    start_time = time.time()
    with Pool(processes=max_processes) as pool:
        for result in tqdm(
            pool.imap_unordered(process_one_image, job_args),
            total=len(reference_files),
        ):
            print(result)
    total_time = time.time() - start_time
    print(f"\n⏱️ Tiempo total de ejecución: {total_time:.2f} segundos")


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Recorte y alineado rápido de VIIRS a imágenes ISS georreferenciadas."
    )
    parser.add_argument(
        "--geo_dir",
        type=str,
        required=True,
        help="Directorio con imágenes ISS georreferenciadas (e.g. *_rect.tiff).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directorio de salida para los recortes VIIRS alineados.",
    )
    parser.add_argument(
        "--viirs_tiff",
        type=str,
        required=False,
        default="/home/raul/planb/VNL_v2_npp_2017_global_vcmslcfg_c202101211500.median.tif",
        help="Ruta al mosaico VIIRS (GeoTIFF).",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        required=True,
        help="ID inicial esperado (coincide con el ID de las imágenes ISS).",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        required=True,
        help="ID final esperado.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=8,
        help="Número de procesos (multiprocessing).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "safe"],
        help="Modo general: 'fast' prioriza velocidad; 'safe' prioriza exactitud.",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        choices=["bbox", "polygon"],
        help="Método de recorte ROI. Por defecto: bbox en 'fast', polygon en 'safe'.",
    )
    parser.add_argument(
        "--align",
        type=str,
        default=None,
        choices=["minimal", "exact"],
        help=(
            "Alineado: 'minimal' (rápido) o 'exact' (misma rejilla que reference). "
            "Por defecto: minimal en 'fast', exact en 'safe'."
        ),
    )
    parser.add_argument(
        "--resampling",
        type=str,
        default=None,
        choices=["nearest", "bilinear", "cubic"],
        help="Método de resampling. Por defecto: nearest en 'fast', bilinear en 'safe'.",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="auto",
        help="Hilos de GDAL para reproyección: entero o 'auto' (=cpu_count()).",
    )
    args = parser.parse_args()

    geo_dir = os.path.abspath(args.geo_dir)
    output_dir = os.path.abspath(args.output_dir)
    viirs_tiff = os.path.abspath(args.viirs_tiff)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(viirs_tiff):
        raise FileNotFoundError(f"No se encontró el TIFF de VIIRS: {viirs_tiff}")

    # Defaults según modo
    if args.mode == "fast":
        clip_mode = args.clip or "bbox"
        align_mode = args.align or "minimal"
        resampling = args.resampling or "nearest"
    else:
        clip_mode = args.clip or "polygon"
        align_mode = args.align or "exact"
        resampling = args.resampling or "bilinear"

    # Hilos GDAL
    if args.threads == "auto":
        gdal_threads = cpu_count()
    else:
        try:
            gdal_threads = int(args.threads)
        except Exception:
            gdal_threads = cpu_count()

    process_timelapse_parallel(
        geo_dir=geo_dir,
        output_dir=output_dir,
        viirs_tiff=viirs_tiff,
        start_id=args.start_id,
        end_id=args.end_id,
        nproc=args.nproc,
        clip_mode=clip_mode,
        align_mode=align_mode,
        resampling=resampling,
        gdal_threads=gdal_threads,
    )


if __name__ == "__main__":
    main()