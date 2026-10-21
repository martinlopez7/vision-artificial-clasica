import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog


VERDE_BAJO = np.array([35, 40, 25])
VERDE_ALTO = np.array([85, 255, 255])

HUE_MAX_COBRE = 16.5              # Hue mediano menor -> cobre; mayor -> dorado
SATURACION_MAX_BIMETAL = 150      # Por debajo de este valor se considera bimetalica
SATURACION_CENTRO_BIMETAL = 145   # Centro más saturado -> 2 Euros; menos -> 1 Euro
SATURACION_MINIMA_HUE = 100       # Solo se usan píxeles con S suficiente para la mediana
UMBRAL_ASPECTO_BILLETE = 1.40     # AR del rectángulo mínimo orientado (invariante al giro)

# Prototipos obtenidos de la calibración (H medio, S media y Lab b medios).
PROTOTIPOS_BILLETES = {
    "Billete 5 EUR": {"h": 26.49, "s": 61.67, "b": 142.7, "valor": 5.0},
    "Billete 10 EUR": {"h": 16.41, "s": 92.41, "b": 149.8, "valor": 10.0},
    "Billete 20 EUR": {"h": 38.40, "s": 42.35, "b": 137.2, "valor": 20.0},
}


def calcular_hue_mediana(hsv, contorno, bbox, saturacion_minima=SATURACION_MINIMA_HUE):
    """Hue mediano de los píxeles suficientemente saturados del objeto."""
    x, y, w, h = bbox
    mascara = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mascara, [contorno], -1, 255, cv2.FILLED, offset=(-x, -y))
    pixeles = hsv[y:y+h, x:x+w][mascara > 0]
    if len(pixeles) == 0:
        return 0.0
    validos = pixeles[pixeles[:, 1] >= saturacion_minima]
    if len(validos) == 0:
        validos = pixeles
    return float(np.median(validos[:, 0]))


def calcular_lab_b(imagen_bgr, contorno, bbox):
    """Media del canal b de Lab dentro del contorno del billete."""
    x, y, w, h = bbox
    mascara = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mascara, [contorno], -1, 255, cv2.FILLED, offset=(-x, -y))
    roi_lab = cv2.cvtColor(imagen_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2LAB)
    return float(cv2.mean(roi_lab[:, :, 2], mask=mascara)[0])


def clasificar_billete(hue_medio, saturacion_media, lab_b):
    """Clasifica el billete por mínima distancia a los prototipos calibrados."""
    mejor_nombre = "Billete desconocido"
    mejor_valor = 0.0
    mejor_distancia = float("inf")

    for nombre, prototipo in PROTOTIPOS_BILLETES.items():
        distancia = (
            ((hue_medio - prototipo["h"]) / 180.0) ** 2
            + ((saturacion_media - prototipo["s"]) / 255.0) ** 2
            + ((lab_b - prototipo["b"]) / 255.0) ** 2
        )
        if distancia < mejor_distancia:
            mejor_distancia = distancia
            mejor_nombre = nombre
            mejor_valor = prototipo["valor"]

    return mejor_nombre, mejor_valor


def clasificar_moneda(hue_mediana, saturacion_media, radio):
    """Clasifica una moneda por color (hue mediano) y radio."""
    if saturacion_media < SATURACION_MAX_BIMETAL:
        return "Bimetal (no analizada)", 0.0

    if hue_mediana < HUE_MAX_COBRE:
        if radio < 75.0:
            return "1 Centimo", 0.01
        if radio < 85.0:
            return "2 Centimos", 0.02
        return "5 Centimos", 0.05

    if radio < 89.0:
        return "10 Centimos", 0.10
    if radio < 99.0:
        return "20 Centimos", 0.20
    return "50 Centimos", 0.50


def clasificar_moneda_bimetalica(crop_moneda_bgr, umbral_saturacion):
    """
    Analiza la región interna (40% del radio) para diferenciar 1 Euro de 2 Euros.
    - Centro Dorado (Saturación alta) -> 2 Euros
    - Centro Plateado (Saturación baja) -> 1 Euro
    """
    hsv = cv2.cvtColor(crop_moneda_bgr, cv2.COLOR_BGR2HSV)
    h, w, _ = crop_moneda_bgr.shape

    cx, cy = w // 2, h // 2
    radio_max = min(cx, cy)
    radio_interno = int(radio_max * 0.40)  # Círculo al 40% del centro

    mask_centro = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_centro, (cx, cy), radio_interno, 255, -1)

    canal_s = hsv[:, :, 1]
    saturacion_media_centro = cv2.mean(canal_s, mask=mask_centro)[0]

    if saturacion_media_centro > umbral_saturacion:
        return "2 Euros", 2.00
    else:
        return "1 Euro", 1.00


def seleccionar_imagen(titulo="Selecciona una imagen"):
    """Abre un diálogo para elegir un archivo JPEG."""
    root = tk.Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(
        title=titulo,
        filetypes=[("JPEG", "*.jpg *.jpeg"), ("Todos los archivos", "*.*")]
    )
    try:
        root.destroy()
    except tk.TclError:
        pass
    return ruta


def redimensionar_para_mostrar(imagen, max_ancho=1600, max_alto=1200):
    """Escala la imagen para que quepa en pantalla manteniendo el aspecto."""
    h, w = imagen.shape[:2]
    escala = min(max_ancho / w, max_alto / h, 1.0)
    nuevo_w = int(w * escala)
    nuevo_h = int(h * escala)
    return cv2.resize(imagen, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA), escala

def main():
    print("=" * 60)
    print("SISTEMA DE VISIÓN CLÁSICA - PROCESAMIENTO POR FOTO")
    print("=" * 60)

    # Seleccionar imagen a procesar
    ruta_imagen = seleccionar_imagen("Selecciona la foto a procesar")
    if not ruta_imagen:
        print("No se seleccionó ninguna imagen. Saliendo...")
        return

    frame = cv2.imread(ruta_imagen)
    if frame is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return

    h, w = frame.shape[:2]
    print(f"\nImagen a procesar: {ruta_imagen}")
    print(f"Resolución: {w}x{h}")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # =========================================================================
    # PASO 1: RANGOS DEL TAPETE VERDE (Ajustar si cambia la luz o la tela)
    # =========================================================================
    mask_tapete = cv2.inRange(hsv, VERDE_BAJO, VERDE_ALTO)
    mask_objetos = cv2.bitwise_not(mask_tapete)  # Los objetos pasan a ser blanco (255)

    # Limpieza morfológica para quitar pequeñas motas o ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_OPEN, kernel)
    mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mask_objetos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_acumulado = 0.0
    objetos_procesados = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 2000:  # Filtrar ruido pequeño (adaptado a 4K)
            continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        # Ignorar regiones de fondo que llegan al borde de la fotografía.
        if x == 0 or y == 0 or x + w_box >= frame.shape[1] or y + h_box >= frame.shape[0]:
            continue

        perimetro = cv2.arcLength(cnt, True)
        if perimetro == 0:
            continue

        circularidad = (4 * np.pi * area) / (perimetro ** 2)

        # Rectángulo mínimo orientado: sus lados son el largo y el ancho reales
        # del objeto, independientemente de cómo esté girado sobre el tapete.
        (_, _), (lado_a, lado_b), _ = cv2.minAreaRect(cnt)
        lado_mayor = max(lado_a, lado_b)
        lado_menor = min(lado_a, lado_b)
        aspecto_orientado = lado_mayor / lado_menor if lado_menor > 0 else 0.0

        # Calcular H solo dentro del contorno, sin incluir el tapete del
        # rectángulo que lo rodea.
        roi_hsv = hsv[y:y+h_box, x:x+w_box]
        mascara_contorno = np.zeros((h_box, w_box), dtype=np.uint8)
        cv2.drawContours(
            mascara_contorno, [cnt], -1, 255, -1, offset=(-x, -y)
        )
        hue_medio, saturacion_media, valor_medio = cv2.mean(
            roi_hsv, mask=mascara_contorno
        )[:3]
        hue_mediana = calcular_hue_mediana(hsv, cnt, (x, y, w_box, h_box))

        clase = "Desconocido"
        valor = 0.0

        # =========================================================================
        # PASO 2: ÁRBOL DE DECISIÓN - RAMA A: BILLETES
        # (rectangulares, grandes y en cualquier orientación)
        # =========================================================================
        es_billete = aspecto_orientado > UMBRAL_ASPECTO_BILLETE and area > 100000
        if es_billete:
            lab_b = calcular_lab_b(frame, cnt, (x, y, w_box, h_box))
            clase, valor = clasificar_billete(hue_medio, saturacion_media, lab_b)

        # =========================================================================
        # PASO 3: ÁRBOL DE DECISIÓN - RAMA B: MONEDAS (Forma Circular)
        # =========================================================================
        elif circularidad > 0.75:
            radio = np.sqrt(area / np.pi)

            if saturacion_media < SATURACION_MAX_BIMETAL:
                roi_moneda_bgr = frame[y:y+h_box, x:x+w_box]
                if roi_moneda_bgr.shape[0] > 10 and roi_moneda_bgr.shape[1] > 10:
                    clase, valor = clasificar_moneda_bimetalica(
                        roi_moneda_bgr, SATURACION_CENTRO_BIMETAL
                    )
                else:
                    clase, valor = "Bimetal (no analizada)", 0.0
            else:
                clase, valor = clasificar_moneda(hue_mediana, saturacion_media, radio)

        total_acumulado += valor
        objetos_procesados.append({
            "clase": clase,
            "valor": valor,
            "bbox": (x, y, w_box, h_box),
            "hue_medio": hue_medio,
            "hue_mediana": hue_mediana,
            "saturacion_media": saturacion_media,
            "contorno": cnt
        })

        # Dibujar contorno y texto sobre la imagen original
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
        if y >= 50:
            y_hue = y - 35
            y_clase = y - 10
        else:
            y_hue = y + h_box + 25
            y_clase = y + h_box + 50

        etiqueta_hue = f"H mean: {hue_medio:.1f}" if es_billete else f"H med: {hue_mediana:.1f}"
        cv2.putText(frame, f"{etiqueta_hue} S: {saturacion_media:.0f}", (x, y_hue),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"{clase}", (x, y_clase),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Mostrar importe total monetario
    cv2.rectangle(frame, (20, 20), (900, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"TOTAL: {total_acumulado:.2f} EUR", (40, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

    # Guardar imagen de resultado
    base, ext = os.path.splitext(ruta_imagen)
    ruta_salida = f"{base}_resultado{ext}"
    cv2.imwrite(ruta_salida, frame)
    print(f"\nImagen resultado guardada en: {ruta_salida}")

    # Imprimir resumen en consola
    print("\n--- RESUMEN DE OBJETOS DETECTADOS ---")
    for obj in objetos_procesados:
        print(f"{obj['clase']:20s} -> {obj['valor']:.2f} EUR")
    print(f"{'TOTAL':20s} -> {total_acumulado:.2f} EUR")
    print("-------------------------------------")

    # Mostrar resultado en pantalla (redimensionado)
    frame_mostrar, _ = redimensionar_para_mostrar(frame)
    cv2.imshow("Sistema de Vision Clasica - Monedas y Billetes", frame_mostrar)
    print("\nPulsa 'q' para cerrar la ventana.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
