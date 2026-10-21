import cv2
import numpy as np
import json
import os
import tkinter as tk
from tkinter import filedialog

# Variables globales para la interacción
punto_clic = None
factor_escala = 1.0
objetos_detectados = []  # Lista de dicts con 'contorno' (coord. originales) y 'datos'
imagen_hsv = None
ruta_imagen = ""


def seleccionar_imagen():
    """Abre un diálogo para elegir un archivo JPEG."""
    root = tk.Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen JPEG (4096x3072)",
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


def calcular_hue_mediana(hsv, contorno, bbox, saturacion_minima=100):
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


def detectar_objetos(imagen_bgr):
    """Segmenta el tapete verde, detecta contornos y extrae métricas."""
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    # Segmentar tapete verde (todo lo que NO sea verde son objetos)
    # El tapete cambia de tono y luminosidad por las sombras de la foto.
    # Este rango ampliado incluye también las zonas oscuras del tapete.
    verde_bajo = np.array([35, 40, 25])
    verde_alto = np.array([85, 255, 255])
    mask_tapete = cv2.inRange(hsv, verde_bajo, verde_alto)
    mask_objetos = cv2.bitwise_not(mask_tapete)

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_OPEN, kernel)
    mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mask_objetos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultados = []
    id_obj = 1
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 2000:  # Umbral adaptado a 4K (equivalente a ~500 px en Full HD)
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # Ignorar regiones de fondo que llegan al borde de la fotografía.
        if x == 0 or y == 0 or x + w >= imagen_bgr.shape[1] or y + h >= imagen_bgr.shape[0]:
            continue

        perimetro = cv2.arcLength(cnt, True)
        circularidad = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
        aspect_ratio = float(w) / h if h != 0 else 0

        # Rectángulo mínimo orientado: no depende del giro del objeto.
        (_, _), (lado_a, lado_b), _ = cv2.minAreaRect(cnt)
        aspecto_orientado = (
            max(lado_a, lado_b) / min(lado_a, lado_b) if min(lado_a, lado_b) > 0 else 0
        )

        radio = np.sqrt(area / np.pi)

        # HSV medio calculado solo dentro del contorno
        roi_hsv = hsv[y:y+h, x:x+w]
        mascara_contorno = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mascara_contorno, [cnt], -1, 255, cv2.FILLED, offset=(-x, -y))
        hue_medio, saturacion_media, valor_medio = cv2.mean(roi_hsv, mask=mascara_contorno)[:3]
        hue_mediana = calcular_hue_mediana(hsv, cnt, (x, y, w, h))

        # Clasificación preliminar solo para mostrar en pantalla
        if circularidad > 0.75 and 0.8 <= aspect_ratio <= 1.25:
            tipo = "moneda"
        else:
            tipo = "billete"

        # Datos adicionales para calibrar bimetalicas (centro) y billetes (Lab).
        if tipo == "moneda":
            radio_max = min(w, h) / 2.0
            radio_centro = int(radio_max * 0.40)
            mascara_centro = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mascara_centro, (w // 2, h // 2), radio_centro, 255, -1)
            hue_centro, saturacion_centro, valor_centro = cv2.mean(
                roi_hsv, mask=mascara_centro
            )[:3]
        else:
            roi_lab = cv2.cvtColor(imagen_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2LAB)
            lab_l, lab_a, lab_b = cv2.mean(roi_lab, mask=mascara_contorno)[:3]

        datos = {
            "id": id_obj,
            "tipo": tipo,
            "hue_medio": float(hue_medio),
            "hue_mediana": float(hue_mediana),
            "saturacion_media": float(saturacion_media),
            "valor_medio": float(valor_medio),
            "radio_px": float(radio),
            "area_px2": float(area),
            "bbox": [int(x), int(y), int(w), int(h)],
            "aspect_ratio": float(aspect_ratio),
            "aspecto_orientado": float(aspecto_orientado),
            "circularidad": float(circularidad)
        }
        if tipo == "moneda":
            datos["hue_centro"] = float(hue_centro)
            datos["saturacion_centro"] = float(saturacion_centro)
            datos["valor_centro"] = float(valor_centro)
        else:
            datos["lab_l"] = float(lab_l)
            datos["lab_a"] = float(lab_a)
            datos["lab_b"] = float(lab_b)
        resultados.append({"contorno": cnt, "datos": datos})
        id_obj += 1

    return resultados, hsv


def dibujar_en_imagen(imagen_bgr, objetos):
    """Dibuja contornos y etiquetas sobre la imagen original."""
    for obj in objetos:
        cnt = obj["contorno"]
        datos = obj["datos"]
        x, y, w, h = datos["bbox"]

        cv2.drawContours(imagen_bgr, [cnt], -1, (0, 255, 0), 3)

        texto = f"{datos['id']}: {datos['tipo']} R:{datos['radio_px']:.1f}px Hmed:{datos['hue_mediana']:.1f}"
        y_texto = y - 15 if y > 40 else y + h + 30
        cv2.putText(imagen_bgr, texto, (x, y_texto),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    return imagen_bgr


def capturar_clic(event, x, y, flags, param):
    global punto_clic
    if event == cv2.EVENT_LBUTTONDOWN:
        punto_clic = (x, y)


def main():
    global punto_clic, factor_escala, objetos_detectados, imagen_hsv, ruta_imagen

    ruta_imagen = seleccionar_imagen()
    if not ruta_imagen:
        print("No se seleccionó ninguna imagen. Saliendo...")
        return

    imagen_original = cv2.imread(ruta_imagen)
    if imagen_original is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return

    h, w = imagen_original.shape[:2]
    print("\n" + "=" * 60)
    print("CALIBRACIÓN POR FOTO")
    print("=" * 60)
    print(f"Imagen cargada: {ruta_imagen}")
    print(f"Resolución detectada: {w}x{h}")

    objetos_detectados, imagen_hsv = detectar_objetos(imagen_original)
    print(f"Objetos detectados: {len(objetos_detectados)}")
    mascara_verde = cv2.inRange(
        imagen_hsv, np.array([35, 40, 25]), np.array([85, 255, 255])
    )
    porcentaje_tapete = 100 * cv2.countNonZero(mascara_verde) / mascara_verde.size
    print(f"Píxeles reconocidos como tapete verde: {porcentaje_tapete:.1f}%")
    if porcentaje_tapete < 50:
        print("AVISO: se reconoce poco tapete; revisa los límites HSV de la máscara.")

    # Dibujar sobre una copia de la imagen original y luego escalar para mostrar
    imagen_anotada = dibujar_en_imagen(imagen_original.copy(), objetos_detectados)
    imagen_mostrar, factor_escala = redimensionar_para_mostrar(imagen_anotada)

    cv2.namedWindow("CALIBRACION - Haz clic en los objetos")
    cv2.setMouseCallback("CALIBRACION - Haz clic en los objetos", capturar_clic)

    print("\nINSTRUCCIONES:")
    print("- Revisa en pantalla cada objeto detectado.")
    print("- Haz CLIC IZQUIERDO sobre un objeto para ver sus datos en consola.")
    print("- Pulsa 'q' para salir y guardar calibracion.json.")
    print("=" * 60)

    while True:
        frame = imagen_mostrar.copy()

        if punto_clic is not None:
            cx, cy = punto_clic
            cx_orig = int(cx / factor_escala)
            cy_orig = int(cy / factor_escala)
            pixel_hsv = imagen_hsv[cy_orig, cx_orig]

            encontrado = False
            for obj in objetos_detectados:
                cnt = obj["contorno"]
                datos = obj["datos"]
                if cv2.pointPolygonTest(cnt, (cx_orig, cy_orig), False) >= 0:
                    print(f"\n[OBJETO {datos['id']} - CLIC EN ({cx_orig}, {cy_orig})]")
                    print(f" -> Tipo preliminar       : {datos['tipo']}")
                    print(f" -> HSV medio del objeto  : H={datos['hue_medio']:.2f}, "
                          f"S={datos['saturacion_media']:.2f}, V={datos['valor_medio']:.2f}")
                    print(f" -> Hue mediano (S>100)   : {datos['hue_mediana']:.2f}")
                    print(f" -> HSV píxel del clic    : H={pixel_hsv[0]}, S={pixel_hsv[1]}, V={pixel_hsv[2]}")
                    if datos["tipo"] == "moneda":
                        print(f" -> Radio equivalente     : {datos['radio_px']:.2f} píxeles")
                        print(f" -> HSV centro (40%)      : H={datos['hue_centro']:.2f}, "
                              f"S={datos['saturacion_centro']:.2f}, V={datos['valor_centro']:.2f}")
                    else:
                        print(f" -> Lab medio             : L={datos['lab_l']:.2f}, "
                              f"a={datos['lab_a']:.2f}, b={datos['lab_b']:.2f}")
                    print(f" -> Área                  : {datos['area_px2']:.1f} px²")
                    print(f" -> Bounding Box          : x={datos['bbox'][0]}, y={datos['bbox'][1]}, "
                          f"w={datos['bbox'][2]}, h={datos['bbox'][3]}")
                    print(f" -> Aspect Ratio          : {datos['aspect_ratio']:.2f}")
                    print(f" -> Aspecto orientado     : {datos['aspecto_orientado']:.2f}")
                    print(f" -> Circularidad          : {datos['circularidad']:.3f}")
                    encontrado = True
                    break

            if not encontrado:
                print(f"\n[CLIC EN FONDO/TAPETE ({cx_orig}, {cy_orig})]")
                print(f" -> Color HSV del tapete: H={pixel_hsv[0]}, S={pixel_hsv[1]}, V={pixel_hsv[2]}")

            punto_clic = None

        cv2.imshow("CALIBRACION - Haz clic en los objetos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Guardar calibracion.json en la misma carpeta que este script
    datos_para_json = [obj["datos"] for obj in objetos_detectados]
    salida = {
        "ruta_imagen": ruta_imagen,
        "resolucion": [w, h],
        "num_objetos": len(datos_para_json),
        "objetos": datos_para_json
    }

    ruta_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibracion.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print(f"\nCalibración guardada en: {ruta_json}")
    print(f"Total de objetos guardados: {len(datos_para_json)}")


if __name__ == "__main__":
    main()
