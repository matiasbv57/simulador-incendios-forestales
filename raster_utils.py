# Importa la biblioteca rasterio, esencial para la lectura y manipulación de datos geoespaciales en formato raster (como GeoTIFF).
import rasterio
# Importa la enumeración Resampling de rasterio, que contiene los diferentes algoritmos de remuestreo de píxeles.
from rasterio.enums import Resampling
# Importa la biblioteca NumPy, fundamental para realizar operaciones numéricas eficientes con las matrices (arrays) que representan los rasters.
import numpy as np

def cargar_raster(path, shape=(100, 100)):
    """
    Carga un archivo raster desde una ruta, lo redimensiona a una forma específica y normaliza sus valores.

    Esta función es genérica y útil para pre-procesar capas de entrada como la pendiente o la elevación,
    asegurando que tengan las mismas dimensiones que la grilla de simulación y que sus valores estén en un rango estándar (0 a 1).

    Args:
        path (str): La ruta al archivo raster (ej. "datos/pendiente.tif").
        shape (tuple): Una tupla (filas, columnas) que define las dimensiones de salida del array.
                         Debe coincidir con el `grid_size` de la simulación.

    Returns:
        np.ndarray: Un array de NumPy 2D con los datos del raster, redimensionado y normalizado.
    """
    # El bloque 'with' asegura que el archivo se cierre correctamente después de su uso.
    with rasterio.open(path) as src:
        # Lee la primera banda (banda 1) del archivo raster.
        # 'out_shape' redimensiona el raster al vuelo a la forma deseada (ej. 600x600).
        # 'resampling=Resampling.bilinear' especifica el algoritmo para el redimensionado.
        # La interpolación bilineal es una buena opción para datos continuos (como la pendiente),
        # ya que calcula el valor de cada nuevo píxel basándose en un promedio ponderado de los 4 píxeles más cercanos del original.
        array = src.read(
            1,
            out_shape=(shape[0], shape[1]),
            resampling=Resampling.bilinear
        )
    
    # Los rasters a menudo contienen valores 'NoData' (sin datos) que se leen como 'NaN' (Not a Number).
    # 'np.nan_to_num' reemplaza todas las ocurrencias de NaN con un valor específico, en este caso 0.0.
    # Esto es crucial para evitar errores en cálculos matemáticos posteriores.
    array = np.nan_to_num(array, nan=0.0)
    
    # Se realiza una normalización min-max para escalar todos los valores del array al rango [0, 1].
    # La fórmula es: (valor - mínimo) / (máximo - mínimo).
    # Esto es importante para que diferentes factores (como pendiente y viento) contribuyan de forma ponderada
    # y comparable en la fórmula de probabilidad de propagación del fuego.
    # Se añade una pequeña constante (epsilon) al denominador para evitar la división por cero si todos los valores son iguales.
    min_val = array.min()
    max_val = array.max()
    if (max_val - min_val) > 0:
        array = (array - min_val) / (max_val - min_val)
    else:
        # Si todos los valores son iguales, el array normalizado será todo ceros.
        array = np.zeros(shape)
        
    return array

def cargar_savi(path, shape=(100, 100), umbral=0.2):
    """
    Carga un raster de índice de vegetación (SAVI), lo procesa y lo convierte en una máscara binaria.

    Esta función está especializada para crear la capa base de vegetación. Determina qué celdas
    tienen suficiente biomasa para ser consideradas "combustibles" en la simulación.

    Args:
        path (str): La ruta al archivo raster del SAVI.
        shape (tuple): Las dimensiones de la grilla de simulación.
        umbral (float): El valor de SAVI normalizado por encima del cual una celda se considera vegetación.

    Returns:
        np.ndarray: Un array de NumPy 2D binario (0 o 1). 1 significa presencia de vegetación, 0 significa ausencia.
    """
    # Al igual que en la función anterior, se abre el archivo de forma segura.
    with rasterio.open(path) as src:
        # Lee y redimensiona el raster SAVI usando interpolación bilineal.
        savi = src.read(
            1,
            out_shape=shape,
            resampling=Resampling.bilinear
        )
        
    # Reemplaza los valores 'NoData' por 0.
    savi = np.nan_to_num(savi, nan=0.0)
    
    # Normaliza los valores del SAVI al rango [0, 1] para que el umbral sea consistente.
    min_val = savi.min()
    max_val = savi.max()
    if (max_val - min_val) > 0:
        savi = (savi - min_val) / (max_val - min_val)
    else:
        savi = np.zeros(shape)

    # Este es el paso clave: la binarización.
    # (savi > umbral) crea una matriz booleana (True/False).
    # .astype(int) convierte esa matriz booleana a una matriz de enteros, donde True se convierte en 1 y False en 0.
    # El resultado es una grilla que representa directamente dónde hay vegetación (1) y dónde no (0).
    return (savi > umbral).astype(int)