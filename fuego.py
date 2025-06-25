# fuego.py
# Módulo que contiene la lógica de propagación del fuego y cálculos relacionados al viento.

import numpy as np
import random
from math import radians, cos, sin

# Estados posibles de una celda (coherentes con el main)
VACIO = 0
VEGETACION = 1
FUEGO = 2
QUEMADO = 3

def direccion_vector(grados):
    """
    Convierte una dirección de viento (en grados) a un vector 2D (dx, dy) redondeado.
    """
    rad = radians(grados)
    return round(cos(rad)), round(sin(rad))

def buscar_vecino(grilla, cx, cy, radio=3):
    """
    Busca una celda vecina con vegetación dentro de un radio alrededor de (cx, cy).
    """
    for r in range(1, radio + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grilla.shape[1] and 0 <= ny < grilla.shape[0]:
                    if grilla[ny, nx] == VEGETACION:
                        return nx, ny
    return None, None

def actualizar(grilla, viento, velocidad, pendiente):
    """
    Propaga el fuego a la siguiente hora según vegetación, pendiente y viento.
    """
    nueva = grilla.copy()
    for y in range(grilla.shape[0]):
        for x in range(grilla.shape[1]):
            if grilla[y, x] == FUEGO:
                nueva[y, x] = QUEMADO
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grilla.shape[1] and 0 <= ny < grilla.shape[0]:
                            if grilla[ny, nx] == VEGETACION:
                                pend = pendiente[ny, nx]
                                viento_dx, viento_dy = viento
                                es_viento = (dx == viento_dx and dy == viento_dy)
                                if es_viento:
                                    prob = 0.6 + 0.2 * pend + 0.02 * velocidad
                                else:
                                    prob = 0.3 + 0.1 * pend + 0.01 * velocidad
                                if random.random() < min(prob, 1.0):
                                    nueva[ny, nx] = FUEGO
    return nueva
