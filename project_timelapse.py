#!/usr/bin/env python3
"""
Proyección de píxeles ISS → Tierra usando la simulación de Blender (v3).

Versión pensada para integrarse con la pipeline:

- Ángulos (yaw/pitch/roll) vienen de la pipeline (fijados o de angle_search).
- Parámetros de cámara (focal, sensor_width, sensor_height, pixel_width, pixel_height)
  también vienen de la pipeline (a partir de los EXIF de la primera imagen).
- Rango temporal (start_date / end_date) viene de la pipeline, ya con offsets aplicados.

Para cada instante de tiempo:
1. Busca el CSV correspondiente a ese timestamp.
2. Lee la imagen real correspondiente.
3. Reconstruye la cámara en Blender con los parámetros dados.
4. Traza rayos desde cada píxel simulado hacia la esfera-Tierra.
5. Intersecta con la esfera y genera archivos .points (QGIS) para real y simulada.

Uso típico desde la pipeline:

    python -m scripts_v3.project_timelapse \
        --output_directory ... \
        --texture_path ... \
        --csv_dir ... \
        --image_dir ... \
        --tle_directory ... \
        --yaw ... --pitch ... --roll ... \
        --focal_length ... \
        --sensor_width ... --sensor_height ... \
        --pixel_width ... --pixel_height ... \
        --start_date ... --end_date ... \
        --time_step ... \
        --points_mode real|simulated|both \
        --orientation_mode north|forward
"""

import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2

from .iss_simulation import (
    reset_scene,
    list_tle_files,
    read_tle_from_files,
    find_closest_tle,
    check_tle_validity,
    get_iss_position_and_velocity,
    creaimagen,
    project_pixels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procesar timelapse ISS: proyectar píxeles y generar .points"
    )
    parser.add_argument(
        '--output_directory', type=str, required=True,
        help='Directorio para imágenes renderizadas y archivos .points'
    )
    parser.add_argument(
        '--texture_path', type=str, required=True,
        help='Ruta a la textura de la Tierra (para reset_scene)'
    )
    parser.add_argument(
        '--csv_dir', type=str, required=True,
        help='Directorio con archivos CSV de coordenadas transformadas (match_timelapse)'
    )
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='Directorio con imágenes reales (pics)'
    )
    parser.add_argument(
        '--tle_directory', type=str, required=True,
        help='Directorio con archivos TLE de la ISS'
    )

    # Ángulos de la cámara (ya ajustados o fijos)
    parser.add_argument('--yaw', type=float, required=True, help='Ángulo yaw en grados')
    parser.add_argument('--pitch', type=float, required=True, help='Ángulo pitch en grados')
    parser.add_argument('--roll', type=float, required=True, help='Ángulo roll en grados')

    # Parámetros de la cámara (desde EXIF / pipeline)
    parser.add_argument(
        '--focal_length', type=float, required=True,
        help='Longitud focal (mm)'
    )
    parser.add_argument(
        '--sensor_width', type=float, required=True,
        help='Ancho de sensor (mm)'
    )
    parser.add_argument(
        '--sensor_height', type=float, required=True,
        help='Alto de sensor (mm)'
    )
    parser.add_argument(
        '--pixel_width', type=int, required=True,
        help='Resolución horizontal del render (px)'
    )
    parser.add_argument(
        '--pixel_height', type=int, required=True,
        help='Resolución vertical del render (px)'
    )

    # Rango temporal (ya calculado en la pipeline a partir de EXIF + offset)
    parser.add_argument(
        '--start_date', type=str, required=True,
        help='Fecha inicio UTC en formato ISO 8601 (ej: 2017-09-13T21:44:33+00:00)'
    )
    parser.add_argument(
        '--end_date', type=str, required=True,
        help='Fecha fin UTC en formato ISO 8601'
    )
    parser.add_argument(
        '--time_step', type=float, default=0.5,
        help='Intervalo temporal entre frames (segundos)'
    )

    parser.add_argument(
        '--points_mode', type=str, default='real',
        choices=['real', 'simulated', 'both'],
        help="Qué .points conservar: solo 'real', solo 'simulated' o 'both'"
    )
    parser.add_argument(
        '--orientation_mode', type=str, default='forward',
        choices=['north', 'forward'],
        help="Modo de orientación de la cámara ('north' o 'forward')"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_directory = Path(args.output_directory)
    csv_dir = Path(args.csv_dir)
    image_dir = Path(args.image_dir)
    tle_directory = Path(args.tle_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    # Listar archivos
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    ])

    if not image_files:
        print(f"No se encontraron imágenes en {image_dir}")
        sys.exit(1)

    # Parsear fechas (start/end ya son UTC en la pipeline)
    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    time_step = timedelta(seconds=args.time_step)

    # Reiniciar escena y leer TLEs
    print("🔁 Reiniciando escena de Blender y cargando TLEs...")
    sphere = reset_scene(earth_radius=10, texture_path=args.texture_path)
    file_paths = list_tle_files(str(tle_directory))
    tle_data = read_tle_from_files(file_paths)

    # Parámetros de cámara (ya vienen de la pipeline)
    focal_length  = float(args.focal_length)
    sensor_width  = float(args.sensor_width)
    sensor_height = float(args.sensor_height)
    pixel_width   = int(args.pixel_width)
    pixel_height  = int(args.pixel_height)

    print(f"Usando cámara:")
    print(f"  focal_length = {focal_length} mm")
    print(f"  sensor       = {sensor_width} x {sensor_height} mm")
    print(f"  resolución   = {pixel_width} x {pixel_height} px")
    print(f"Ángulos:")
    print(f"  yaw   = {args.yaw}")
    print(f"  pitch = {args.pitch}")
    print(f"  roll  = {args.roll}")
    print(f"orientation_mode = {args.orientation_mode}")

    current_time = start_date
    image_index = 0

    # Loop temporal: igual filosofía que generate_timelapse
    while current_time <= end_date and image_index < len(image_files):

        # Este timestamp debe coincidir con el que se usó en generate_timelapse
        # para nombrar render_output_... y en match_timelapse para los CSV.
        timestamp_str = current_time.strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]

        # CSV generado por match_timelapse
        expected_prefix = f"transformed_coordinates_render_output_{timestamp_str}"
        csv_file = next(
            (f for f in csv_files if f.startswith(expected_prefix)),
            None
        )

        if csv_file is None:
            print(f"⚠️ No se encontró archivo CSV para {timestamp_str}, saltando frame.")
            current_time += time_step
            image_index += 1
            continue

        csv_path = csv_dir / csv_file
        image_file = image_files[image_index]
        image_path = image_dir / image_file

        real_photo = cv2.imread(str(image_path))
        if real_photo is None:
            print(f"⚠️ No se pudo leer la imagen {image_path}, saltando frame.")
            current_time += time_step
            image_index += 1
            continue

        real_photo_height = real_photo.shape[0]

        print(f"[Procesando] tiempo={timestamp_str}")
        print(f"   CSV:    {csv_file}")
        print(f"   Imagen: {image_file} (altura real = {real_photo_height}px)")

        # TLE más cercano para el instante actual
        closest_tle = find_closest_tle(tle_data, current_time)
        check_tle_validity(closest_tle, current_time)

        latitude, longitude, altitude, v_icrf, v_itrs = get_iss_position_and_velocity(closest_tle, current_time)
        velocity = v_itrs  # para forward (recomendado)

        observation_time_str = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        # Antes de proyectar, anotamos qué .points existen ya,
        # para poder saber cuáles son nuevos y aplicar points_mode.
        before_points = set(
            f for f in os.listdir(output_directory) if f.endswith(".points")
        )

        # Crear la cámara en Blender, SIN renderizar imagen
        camera, _ = creaimagen(
            latitude, longitude, altitude,
            args.yaw, args.pitch, args.roll,
            velocity, sphere,
            focal_length, sensor_width, sensor_height,
            pixel_width, pixel_height,
            observation_time_str, str(output_directory), 10,
            render_image=False,
            orientation_mode=args.orientation_mode,
        )

        # Proyectar píxeles (usa los sim_x/sim_y del CSV y la cámara recién creada)
        project_pixels(
            str(csv_path),
            latitude, longitude, altitude,
            args.yaw, args.pitch, args.roll,
            velocity, sphere, camera,
            real_photo_height, observation_time_str,
            str(output_directory), 10,
        )

        # Después de proyectar, vemos qué .points nuevos se han creado
        after_points = set(
            f for f in os.listdir(output_directory) if f.endswith(".points")
        )
        new_points = after_points - before_points

        # Filtramos según points_mode
        if args.points_mode == "real":
            for fname in new_points:
                if fname.endswith("_simulated.points"):
                    try:
                        os.remove(output_directory / fname)
                    except Exception as e:
                        print(f"⚠️ No se pudo borrar {fname}: {e}")
        elif args.points_mode == "simulated":
            for fname in new_points:
                if fname.endswith("_real.points"):
                    try:
                        os.remove(output_directory / fname)
                    except Exception as e:
                        print(f"⚠️ No se pudo borrar {fname}: {e}")
        # "both": no se borra nada

        print(f"✔ Frame procesado: tiempo={timestamp_str}, CSV={csv_file}, imagen={image_file}")

        current_time += time_step
        image_index += 1

    print("✅ Todos los frames del timelapse han sido procesados.")


if __name__ == "__main__":
    main()
