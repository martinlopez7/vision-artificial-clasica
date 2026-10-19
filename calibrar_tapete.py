import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


# Margen que se añade a los percentiles para cubrir variaciones leves de luz
# y los artefactos de compresión JPEG. Puedes cambiarlo y volver a ejecutar.
PERCENTIL_INFERIOR = 1
PERCENTIL_SUPERIOR = 99
MARGEN_H = 3
MARGEN_SV = 15


def seleccionar_imagen():
    root = tk.Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(
        title="Selecciona una foto que contenga solo el tapete verde",
        filetypes=[("Imágenes JPEG", "*.jpg *.jpeg"), ("Todas las imágenes", "*.png *.bmp *.tif *.tiff")],
    )
    root.destroy()
    return ruta


def redimensionar(imagen, ancho_maximo, alto_maximo):
    alto, ancho = imagen.shape[:2]
    escala = min(ancho_maximo / ancho, alto_maximo / alto, 1.0)
    nuevo_tamano = (max(1, int(ancho * escala)), max(1, int(alto * escala)))
    return cv2.resize(imagen, nuevo_tamano, interpolation=cv2.INTER_AREA)


def calcular_limites(hsv):
    """Calcula límites robustos y devuelve también los percentiles observados."""
    canales = ("H", "S", "V")
    inferiores = []
    superiores = []
    estadisticas = {}

    for indice, nombre in enumerate(canales):
        canal = hsv[:, :, indice]
        p_inferior, p_superior = np.percentile(
            canal, [PERCENTIL_INFERIOR, PERCENTIL_SUPERIOR]
        )
        margen = MARGEN_H if nombre == "H" else MARGEN_SV
        limite_maximo = 179 if nombre == "H" else 255
        inferior = max(0, int(np.floor(p_inferior)) - margen)
        superior = min(limite_maximo, int(np.ceil(p_superior)) + margen)

        inferiores.append(inferior)
        superiores.append(superior)
        estadisticas[nombre] = {
            "min": int(canal.min()),
            "max": int(canal.max()),
            "percentil_inferior": float(p_inferior),
            "percentil_superior": float(p_superior),
        }

    return np.array(inferiores, dtype=np.uint8), np.array(superiores, dtype=np.uint8), estadisticas


def main():
    ruta = seleccionar_imagen()
    if not ruta:
        print("No se seleccionó ninguna imagen.")
        return

    # imdecode/fromfile permite abrir rutas que contienen caracteres no ASCII en Windows.
    try:
        imagen = cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as error:
        print(f"No se pudo leer la imagen: {error}")
        return

    if imagen is None:
        print(f"No se pudo abrir la imagen: {ruta}")
        return

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    limite_inferior, limite_superior, estadisticas = calcular_limites(hsv)
    mascara = cv2.inRange(hsv, limite_inferior, limite_superior)
    porcentaje_incluido = 100.0 * cv2.countNonZero(mascara) / mascara.size

    alto, ancho = imagen.shape[:2]
    print("\nCALIBRACIÓN DEL TAPETE VERDE")
    print("=" * 60)
    print(f"Imagen: {ruta}")
    print(f"Resolución: {ancho}x{alto}")
    print("\nRangos HSV medidos (mínimo y máximo absolutos):")
    for canal, datos in estadisticas.items():
        print(f"  {canal}: {datos['min']} .. {datos['max']}")
    print(f"\nPercentiles usados: {PERCENTIL_INFERIOR} .. {PERCENTIL_SUPERIOR}")
    for canal, datos in estadisticas.items():
        print(
            f"  {canal}: {datos['percentil_inferior']:.1f} .. "
            f"{datos['percentil_superior']:.1f}"
        )

    print("\nLímites sugeridos para copiar al mask_tapete:")
    print(f"verde_bajo = np.array({limite_inferior.tolist()})")
    print(f"verde_alto = np.array({limite_superior.tolist()})")
    print(f"Píxeles incluidos por la máscara sugerida: {porcentaje_incluido:.2f}%")
    print("\nPulsa 'q' o ESC para cerrar la vista previa.")

    vista_ancho = 800
    vista_alto = 600
    imagen_vista = redimensionar(imagen, vista_ancho, vista_alto)
    mascara_vista = redimensionar(mascara, vista_ancho, vista_alto)
    mascara_color = cv2.cvtColor(mascara_vista, cv2.COLOR_GRAY2BGR)
    comparacion = np.hstack((imagen_vista, mascara_color))

    alto_vista, ancho_vista = comparacion.shape[:2]
    cv2.putText(
        comparacion, "Tapete original", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        comparacion, "Mascara sugerida", (vista_ancho + 15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.namedWindow("Calibracion del tapete", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibracion del tapete", ancho_vista, alto_vista)
    cv2.imshow("Calibracion del tapete", comparacion)
    while True:
        tecla = cv2.waitKey(0) & 0xFF
        if tecla in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
