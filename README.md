# simulador-incendios-forestales
Simulador de incendios forestales con viento, SAVI, pendiente y orientación. Aplicado a Chilecito la rioja Autor: Matías D. Tejada - Bomberos Voluntarios.
import pygame
import requests
from math import radians, cos, sin
import raster_utils

def obtener_viento_24h(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "auto"
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise Exception("Error al obtener datos meteorológicos")
    data = resp.json()
    velocidades = data["hourly"]["wind_speed_10m"]
    direcciones = data["hourly"]["wind_direction_10m"]
    return velocidades, direcciones

def direccion_a_vector(dir_grados):
    rad = radians((dir_grados + 180) % 360)  # Viento hacia dónde va
    dx = round(cos(rad), 2)
    dy = round(sin(rad), 2)
    return dx, dy

def main():
    # Coordenadas de inicio (Capilla del Monte)
    lat, lon = -30.85, -64.48
    velocidades, direcciones = obtener_viento_24h(lat, lon)

    # Cargar raster y mapas
    savi = raster_utils.cargar_raster("datos/savi.tif")
    pendiente = raster_utils.cargar_raster("datos/pendiente.tif")
    orientacion = raster_utils.cargar_raster("datos/orientacion.tif")

    # Inicialización visual con Pygame (ejemplo simple)
    pygame.init()
    pantalla = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("Simulador de Incendio – Capilla del Monte")

    # Punto de ignición
    fuego = [(400, 400)]  # coordenadas visuales simuladas

    reloj = pygame.time.Clock()

    hora = 0
    corriendo = True
    while corriendo and hora < len(velocidades):
        pantalla.fill((0, 0, 0))

        # Propagación (simplificada)
        nueva_frontera = []
        for x, y in fuego:
            pygame.draw.circle(pantalla, (255, 0, 0), (x, y), 4)
            dx, dy = direccion_a_vector(direcciones[hora])
            nueva_frontera.append((x + int(dx*5), y - int(dy*5)))  # invertido eje Y
        fuego += nueva_frontera

        # Mostrar hora y viento
        fuente = pygame.font.SysFont(None, 24)
        info = fuente.render(f"Hora {hora}: {velocidades[hora]} m/s, dirección {direcciones[hora]}°", True, (255, 255, 255))
        pantalla.blit(info, (10, 10))

        pygame.display.flip()
        reloj.tick(1)  # 1 cuadro por segundo
        hora += 1

    pygame.quit()

if __name__ == "__main__":
    main()
import rasterio
import numpy as np

def cargar_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)  # Leer banda 1
# 🔥 Simulador de Incendios Forestales

Simulador desarrollado por **Matías D. Tejada**, Bombero Voluntario de Chilecito, para modelar el avance de incendios en tiempo real con condiciones reales.

## Funcionalidades
- Ingreso de coordenadas
- Viento real (dirección e intensidad)
- Cálculo de propagación por hora
- Influencia de SAVI, pendiente y orientación de ladera
- Visualización animada sobre mapa

## Autor
Matías D. Tejada  
Bomberos Voluntarios – Área SIG
__pycache__/
*.pyc
*.tif
*.aux.xml
.DS_Store
.env
