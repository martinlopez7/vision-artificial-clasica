import cv2
import numpy as np

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
    radio_interno = int(radio_max * 0.40) # Círculo al 40% del centro
    
    mask_centro = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_centro, (cx, cy), radio_interno, 255, -1)
    
    canal_s = hsv[:, :, 1]
    saturacion_media_centro = cv2.mean(canal_s, mask=mask_centro)[0]
    
    if saturacion_media_centro > umbral_saturacion:
        return "2_EUROS", 2.00
    else:
        return "1_EURO", 1.00


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # =========================================================================
        # PASO 1: RANGOS DEL TAPETE VERDE (Ajustar si cambia la luz o la tela)
        # =========================================================================
        verde_bajo = np.array([35, 40, 40])
        verde_alto = np.array([85, 255, 255])
        
        mask_tapete = cv2.inRange(hsv, verde_bajo, verde_alto)
        mask_objetos = cv2.bitwise_not(mask_tapete) # Los objetos pasan a ser blanco (255)

        # Limpieza morfológica para quitar pequeñas motas o ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_OPEN, kernel)
        mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mask_objetos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_acumulado = 0.0

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < 500: # Filtrar ruido pequeñito
                continue

            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue

            circularidad = (4 * np.pi * area) / (perimetro ** 2)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            clase = "Desconocido"
            valor = 0.0

            # =========================================================================
            # PASO 2: ABRAT DE DECISIÓN - RAMA A: BILLETES (Rectangulares y grandes)
            # =========================================================================
            if aspect_ratio > 1.25 or aspect_ratio < 0.8:
                roi_billete = hsv[y:y+h, x:x+w]
                hue_medio = cv2.mean(roi_billete[:, :, 0])[0]

                # Tono Hue del Billete
                if 95 <= hue_medio <= 130:
                    clase, valor = "Billete 20 EUR", 20.0
                elif (0 <= hue_medio <= 15) or (165 <= hue_medio <= 180):
                    clase, valor = "Billete 10 EUR", 10.0
                elif 35 <= hue_medio <= 85:
                    clase, valor = "Billete 5 EUR", 5.0

            # =========================================================================
            # PASO 3: ÁRBOL DE DECISIÓN - RAMA B: MONEDAS (Forma Circular)
            # =========================================================================
            elif circularidad > 0.75:
                radio = np.sqrt(area / np.pi)
                roi_moneda_bgr = frame[y:y+h, x:x+w]
                roi_moneda_hsv = hsv[y:y+h, x:x+w]
                hue_medio = cv2.mean(roi_moneda_hsv[:, :, 0])[0]

                # ---------------------------------------------------------------------
                # CÁLCULO DE GRUPOS POR COLOR
                # ---------------------------------------------------------------------
                
                # 1. Grupo Cobrizo (1ct, 2ct, 5ct) -> Tono Naranja/Marrón
                if 0 <= hue_medio <= 25:
                    # ==== MODIFICAR ESTOS UMBRALES DE RADIO (px) CON TU CALIBRACIÓN ====
                    if radio < 25.0:
                        clase, valor = "1 Centimo", 0.01
                    elif radio < 32.0:
                        clase, valor = "2 Centimos", 0.02
                    else:
                        clase, valor = "5 Centimos", 0.05

                # 2. Grupo Dorado (10ct, 20ct, 50ct) -> Tono Amarillo/Dorado
                elif 26 <= hue_medio <= 45:
                    # ==== MODIFICAR ESTOS UMBRALES DE RADIO (px) CON TU CALIBRACIÓN ====
                    if radio < 30.0:
                        clase, valor = "10 Centimos", 0.10
                    elif radio < 36.0:
                        clase, valor = "20 Centimos", 0.20
                    else:
                        clase, valor = "50 Centimos", 0.50

                # 3. Grupo Bimetálico (1€ y 2€) -> Región Interna
                else:
                    if roi_moneda_bgr.shape[0] > 10 and roi_moneda_bgr.shape[1] > 10:
                        # ==== MODIFICAR EL UMBRAL DE SATURACIÓN (Ej. 60) TRAS CALIBRAR ====
                        UMBRAL_SATURACION_CENTRO = 60
                        clase, valor = clasificar_moneda_bimetalica(roi_moneda_bgr, UMBRAL_SATURACION_CENTRO)

            total_acumulado += valor

            # Dibujar contorno y texto sobre la webcam
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(frame, f"{clase}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Mostrar Nivel 3: Interpretación (Importe Total Monetario)
        cv2.rectangle(frame, (10, 10), (450, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL: {total_acumulado:.2f} EUR", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.imshow("Sistema de Vision Clasica - Monedas y Billetes", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()