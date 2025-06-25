# ======================================================
# Simulador de Incendios Forestales - 
# ------------------------------------------------------
# Versión extendida con:
# - Integración API Open-Meteo para viento real (48 hs)
# - Visualización de fondo OSM (georreferenciado)
# - Propagación del fuego según viento, pendiente y vegetación (SAVI)
# - Exportación de resultados en SHP, MP4, y GIF
# - Exportación de punto de inicio del fuego como SHP
# ======================================================

# --- IMPORTACIÓN DE BIBLIOTECAS ---
# Se importan todas las librerías necesarias para el funcionamiento del simulador.
import pygame  # Para la visualización gráfica y la interfaz de usuario.
import numpy as np  # Para la manipulación eficiente de la grilla y cálculos numéricos.
import random  # Utilizada para procesos estocásticos (no se usa en esta versión pero es útil para extensiones).
import sys  # Para gestionar la salida del programa.
import rasterio  # Fundamental para leer y procesar datos geoespaciales en formato raster (TIF).
import requests  # Para realizar peticiones a la API de Open-Meteo y obtener datos del viento.
import imageio  # Para crear el archivo GIF animado a partir de los fotogramas guardados.
import os  # Para interactuar con el sistema operativo, como crear directorios.
import geopandas as gpd  # Para manejar datos geoespaciales vectoriales y exportar a Shapefile (SHP).
import cv2  # (OpenCV) Para procesar los fotogramas y exportar el video en formato MP4.
from shapely.geometry import Polygon, Point  # Para crear geometrías (polígonos, puntos) que se usarán con Geopandas.
from raster_utils import cargar_raster, cargar_savi  # Módulos auxiliares para cargar y pre-procesar los datos raster.
from PIL import Image  # (Pillow) Para manipular imágenes, como el mapa de fondo.
from rasterio.transform import from_bounds  # Para crear la transformación afín desde los límites del raster.
from fuego import direccion_vector, buscar_vecino, actualizar  # Módulo con la lógica principal de la propagación del fuego.

# =============== PARÁMETROS GLOBALES DE LA SIMULACIÓN ==================
# Define las dimensiones de la grilla de simulación. Una grilla más grande implica más detalle pero mayor coste computacional.
grid_size = 600
# Define el tamaño en píxeles de cada celda en la ventana de Pygame.
cell_size = 1
# Calcula el ancho y alto de la ventana de la simulación en píxeles.
width, height = grid_size * cell_size, grid_size * cell_size
# Fotogramas por segundo para la visualización (aunque en este bucle se controla por tiempo de espera).
fps = 10

# =============== ESTADOS POSIBLES DE UNA CELDA ===============
# Se definen constantes numéricas para representar los diferentes estados que puede tener una celda en la grilla.
VACIO = 0       # La celda no tiene vegetación combustible.
VEGETACION = 1  # La celda tiene vegetación y puede quemarse.
FUEGO = 2       # La celda está actualmente en llamas.
QUEMADO = 3     # La celda ya se ha quemado y no puede volver a incendiarse.

# =============== PALETA DE COLORES PARA VISUALIZACIÓN (RGBA) ===============
# Un diccionario que asocia cada estado de celda con un color específico para su representación visual.
# Los colores están en formato RGBA (Rojo, Verde, Azul, Alfa/Transparencia).
COLORES = {
    VACIO: (255, 255, 255),          # Blanco para celdas vacías (no se dibuja).
    VEGETACION: (34, 139, 34, 25),   # Verde semitransparente para la vegetación.
    FUEGO: (255, 0, 0, 255),         # Rojo sólido para el fuego.
    QUEMADO: (255, 165, 0, 255)      # Naranja sólido para el área quemada.
}

# =============== CARGA DE DATOS GEOGRÁFICOS =====================
# Carga el raster de pendientes y lo redimensiona para que coincida con la grilla de la simulación.
# La pendiente es un factor clave en la propagación del fuego.
pendiente = cargar_raster("datos/pendiente_capilla.tif", shape=(grid_size, grid_size))


def cargar_mapa_base_osm():
    """
    Carga, redimensiona y convierte la imagen de fondo de OpenStreetMap (OSM)
    para ser usada como base visual en Pygame.
    """
    # Abre la imagen del mapa base.
    img = Image.open("datos/capilla_fondo.png")
    # Redimensiona la imagen para que coincida con el tamaño de la ventana de simulación.
    img = img.resize((width, height))
    # Convierte la imagen al formato RGB, que es compatible con Pygame.
    img = img.convert("RGB")
    # Convierte la imagen de PIL a una superficie de Pygame.
    return pygame.image.fromstring(img.tobytes(), img.size, img.mode)

# Carga la imagen de fondo al iniciar el script para tenerla disponible.
imagen_fondo = cargar_mapa_base_osm()


def generar_grilla():
    """
    Crea la grilla inicial de la simulación a partir del raster SAVI (Índice de Vegetación Ajustado al Suelo).
    Las celdas se clasifican como VEGETACION o VACIO según un umbral.
    """
    # Carga el raster SAVI. Las áreas con un valor SAVI superior al umbral se consideran vegetación.
    vegetacion = cargar_savi("datos/savi_capilla.tif", shape=(grid_size, grid_size), umbral=0.65)
    # Crea una grilla inicial donde todas las celdas son VACIO.
    grilla = np.full((grid_size, grid_size), VACIO)
    # Asigna el estado VEGETACION a las celdas donde el mapa SAVI indica presencia de vegetación.
    grilla[vegetacion == 1] = VEGETACION
    return grilla


def latlon_a_indices(lat, lon, path_tif, grid_size):
    """
    Convierte coordenadas geográficas (latitud, longitud) a índices de la grilla (fila, columna).
    """
    # Abre el archivo raster de referencia para obtener sus metadatos de georreferenciación.
    with rasterio.open(path_tif) as src:
        # Convierte las coordenadas del mundo real (lon, lat) a coordenadas de píxel del raster original.
        row, col = src.index(lon, lat)
        # Obtiene las dimensiones totales del raster original.
        rows, cols = src.shape
    # Escala los índices del raster original a los índices de la grilla de simulación.
    y = int((row / rows) * grid_size)
    x = int((col / cols) * grid_size)
    return x, y


def viento_24h(lat, lon):
    """
    Obtiene el pronóstico de velocidad y dirección del viento para las próximas 48 horas
    desde la API de Open-Meteo para una latitud y longitud específicas.
    """
    # Construye la URL de la API con las coordenadas y los parámetros deseados (viento horario a 10m).
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=winddirection_10m,windspeed_10m&timezone=America%2FSao_Paulo"
    try:
        # Realiza la petición GET a la API.
        resp = requests.get(url)
        # Si la petición fue exitosa (código 200), procesa los datos.
        if resp.status_code == 200:
            data = resp.json()
            # Extrae las listas de dirección y velocidad del viento para las primeras 48 horas.
            direcciones = data["hourly"]["winddirection_10m"][:48]
            velocidades = data["hourly"]["windspeed_10m"][:48]
            return velocidades, direcciones
        else:
            # Si hay un error en la API, informa y devuelve valores por defecto.
            print("⚠️ Error al conectar con Open-Meteo. Usando datos por defecto.")
            return [5] * 48, [90] * 48  # Viento constante de 5 km/h desde el Este (90°).
    except requests.exceptions.ConnectionError:
        # Si hay un error de conexión de red, informa y devuelve valores por defecto.
        print("⚠️ Error de conexión. Verifique su red. Usando datos por defecto.")
        return [5] * 48, [90] * 48


def exportar_shapefile(grilla):
    """
    Exporta el área total quemada (celdas con estado QUEMADO) como un archivo Shapefile (SHP).
    """
    # Abre el raster original para obtener sus metadatos geográficos (límites y sistema de coordenadas).
    with rasterio.open("datos/savi_capilla.tif") as src:
        bounds = src.bounds
        crs = src.crs

    # Crea una transformación afín que mapea los índices de la grilla (0, 600) a coordenadas del mundo real.
    transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, grilla.shape[1], grilla.shape[0])

    celdas_quemadas = []
    # Itera sobre cada celda de la grilla final.
    for y in range(grilla.shape[0]):
        for x in range(grilla.shape[1]):
            # Si una celda está QUEMADA...
            if grilla[y, x] == QUEMADO:
                # ...transforma sus coordenadas de grilla a coordenadas geográficas para definir un polígono.
                x0, y0 = transform * (x, y)
                x1, y1 = transform * (x + 1, y + 1)
                poligono = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                celdas_quemadas.append(poligono)

    # Si no se encontraron celdas quemadas, no se exporta nada.
    if not celdas_quemadas:
        print("⚠️ No hay celdas quemadas para exportar.")
        return

    # Une todos los polígonos individuales en una única geometría (multi-polígono).
    area_total = gpd.GeoSeries(celdas_quemadas).unary_union
    # Crea un GeoDataFrame de Geopandas con la geometría unificada y el sistema de coordenadas correcto.
    gdf = gpd.GeoDataFrame(geometry=[area_total], crs=crs)
    # Guarda el GeoDataFrame como un archivo Shapefile.
    gdf.to_file("outputs/incendio_quemado.shp")
    print("✅ Shapefile del área quemada exportado correctamente.")


def exportar_video():
    """
    Crea un video MP4 a partir de los fotogramas (frames) guardados durante la simulación.
    """
    # Obtiene una lista ordenada de todos los archivos de imagen guardados en la carpeta de frames.
    frame_files = sorted([f"outputs/frames/{f}" for f in os.listdir("outputs/frames") if f.endswith(".png")])
    # Si no hay frames, no hace nada.
    if not frame_files:
        return
    # Lee el primer frame para obtener las dimensiones del video.
    frame = cv2.imread(frame_files[0])
    height, width, _ = frame.shape
    # Inicializa el objeto para escribir el video (formato MP4, 2 fotogramas por segundo, dimensiones).
    out = cv2.VideoWriter("outputs/simulacion_incendio.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 2, (width, height))
    # Itera sobre cada archivo de frame y lo escribe en el video.
    for f in frame_files:
        out.write(cv2.imread(f))
    # Libera el objeto de video, finalizando el proceso de escritura.
    out.release()


def main():
    """
    Función principal que orquesta toda la simulación.
    """
    # Genera la grilla inicial basada en la vegetación.
    grilla = generar_grilla()
    # Pide al usuario que ingrese las coordenadas de inicio del fuego.
    print("Ingresá coordenadas LAT,LON para iniciar fuego (ej: -30.86,-64.53):")
    coord = input(">> ").strip()

    try:
        # Intenta convertir la entrada del usuario a dos números flotantes (latitud y longitud).
        lat, lon = map(float, coord.split(","))
        # Convierte las coordenadas geográficas a índices de la grilla.
        cx, cy = latlon_a_indices(lat, lon, "datos/savi_capilla.tif", grid_size)
    except:
        # Si las coordenadas son inválidas, usa una ubicación por defecto en Capilla del Monte.
        print("Coordenadas inválidas. Usando por defecto chilecito la rioja, Córdoba.")
        cx, cy = grid_size // 2, grid_size // 2
        lat, lon = -29.116495309469578, -67.61160108079707

    # Obtiene los datos de viento para las coordenadas de inicio.
    velocidades, direcciones = viento_24h(lat, lon)

    # Crea un punto geográfico para el inicio del fuego y lo exporta como un Shapefile.
    punto_inicio = Point(lon, lat)
    gdf_inicio = gpd.GeoDataFrame({'evento': ['Inicio_Fuego']}, geometry=[punto_inicio], crs="EPSG:4326")
    gdf_inicio.to_file("outputs/punto_inicio_fuego.shp")
    print("📍 Punto de inicio exportado como SHP.")

    # Inicializa Pygame.
    pygame.init()
    # Crea la ventana de visualización.
    pantalla = pygame.display.set_mode((width, height))
    # Establece el título de la ventana.
    pygame.display.set_caption("🔥 Predicción de Incendios Forestales")
    # Carga una fuente para mostrar texto en pantalla.
    font = pygame.font.SysFont(None, 28)

    # Crea el directorio para guardar los fotogramas si no existe.
    os.makedirs("outputs/frames", exist_ok=True)
    # Lista para almacenar las imágenes que formarán el GIF.
    imagenes = []

    # Bucle principal de la simulación: se ejecuta una vez por cada hora del pronóstico (48 horas).
    for hora in range(48):
        # Obtiene la dirección y velocidad del viento para la hora actual.
        viento_dir = direcciones[hora]
        viento_vel = velocidades[hora]
        # Convierte la dirección del viento (grados) a un vector de propagación (x, y).
        viento = direccion_vector(viento_dir)

        # Manejo de eventos de Pygame (cerrar ventana, clic del mouse).
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            # Si se hace clic con el mouse, se usa esa posición para iniciar un nuevo fuego.
            if e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                cx, cy = mx // cell_size, my // cell_size

        # Lógica para iniciar el fuego.
        if cx is not None:  # Si hay una coordenada de inicio (por input o por clic).
            if grilla[cy, cx] == VEGETACION:
                # Si la celda seleccionada tiene vegetación, se enciende.
                grilla[cy, cx] = FUEGO
            else:
                # Si no, busca la celda con vegetación más cercana en un radio pequeño.
                nx, ny = buscar_vecino(grilla, cx, cy, radio=3)
                if nx is not None:
                    grilla[ny, nx] = FUEGO
            # Resetea las coordenadas de inicio para que el fuego no se reinicie en cada paso.
            cx, cy = None, None

        # --- Actualización del estado de la simulación ---
        # Llama a la función que contiene la lógica de propagación del fuego.
        grilla = actualizar(grilla, viento, viento_vel, pendiente)

        # --- Dibujado en pantalla ---
        # Dibuja el mapa base de OSM.
        pantalla.blit(imagen_fondo, (0, 0))
        # Crea una superficie transparente para dibujar la capa de vegetación.
        savi_layer = pygame.Surface((width, height), pygame.SRCALPHA)

        # Itera sobre cada celda para dibujarla según su estado.
        for y in range(grid_size):
            for x in range(grid_size):
                estado = grilla[y, x]
                if estado == VEGETACION:
                    # Dibuja la vegetación en su capa semitransparente.
                    pygame.draw.rect(savi_layer, COLORES[VEGETACION], (x, y, 1, 1))
                elif estado == FUEGO:
                    # Dibuja el fuego directamente sobre la pantalla principal.
                    pygame.draw.rect(pantalla, COLORES[FUEGO], (x, y, 1, 1))
                elif estado == QUEMADO:
                    # Dibuja el área quemada directamente sobre la pantalla.
                    pygame.draw.rect(pantalla, COLORES[QUEMADO], (x, y, 1, 1))

        # Superpone la capa de vegetación sobre el mapa de fondo.
        pantalla.blit(savi_layer, (0, 0))

        # Dibuja la información de texto en pantalla (hora y datos del viento).
        texto = font.render(f"Hora: {hora}", True, (255, 255, 255))
        pantalla.blit(texto, (10, 10))

        info_viento = font.render(f"Viento: {round(viento_vel)} km/h - {round(viento_dir)}°", True, (255, 255, 255))
        pantalla.blit(info_viento, (width - 290, 10))

        # Dibuja una flecha para indicar la dirección e intensidad del viento.
        vx, vy = direccion_vector(viento_dir)
        longitud = int(viento_vel * 3) # La longitud de la flecha es proporcional a la velocidad.
        inicio = (width - 120, 60)
        fin = (inicio[0] + vx * longitud, inicio[1] - vy * longitud)
        pygame.draw.line(pantalla, (0, 191, 255), inicio, fin, 3) # Dibuja la línea de la flecha.
        pygame.draw.circle(pantalla, (0, 191, 255), inicio, 5) # Dibuja un círculo en el origen de la flecha.


        # Actualiza la pantalla para mostrar todo lo que se ha dibujado.
        pygame.display.flip()

        # Guarda el fotograma actual como una imagen PNG.
        path_img = f"outputs/frames/frame_{hora:02d}.png"
        pygame.image.save(pantalla, path_img)
        # Añade la imagen a la lista para el GIF.
        imagenes.append(imageio.imread(path_img))
        # Pequeña pausa para que la simulación no vaya demasiado rápido.
        pygame.time.wait(500)

    # --- Finalización y exportación de resultados ---
    print("Exportando resultados...")
    # Crea y guarda el GIF animado.
    imageio.mimsave("outputs/simulacion_incendio.gif", imagenes, duration=1)
    # Exporta el área quemada como Shapefile.
    exportar_shapefile(grilla)
    # Exporta la simulación como video MP4.
    exportar_video()
    print("✅ GIF, SHP y MP4 generados correctamente en la carpeta /outputs.")

    # Cierra Pygame y termina el script.
    pygame.quit()
    sys.exit()


# Este bloque asegura que la función main() solo se ejecute cuando el script se corre directamente.
if __name__ == "__main__":
    main()
