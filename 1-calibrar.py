import cv2
import numpy as np

# Variable global para guardar la coordenada del clic
punto_clic = None

def capturar_clic(event, x, y, flags, param):
    global punto_clic
    if event == cv2.EVENT_LBUTTONDOWN:
        punto_clic = (x, y)

# Inicializar webcam (si tienes varias cámaras, cambia el 0 por 1 o 2)
cap = cv2.VideoCapture(0)

# Ajustar resolución a 1080p (Full HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("CALIBRACION - Haz clic en las monedas/billetes")
cv2.setMouseCallback("CALIBRACION - Haz clic en las monedas/billetes", capturar_clic)

print("=" * 60)
print("INSTRUCCIONES DE CALIBRACIÓN:")
print("1. Pon objetos de prueba dentro de la caja sobre el tapete verde.")
print("2. Haz CLIC IZQUIERDO sobre una moneda/billete.")
print("3. Revisa los datos (Radio en px, Tono H, Saturación S) en esta consola.")
print("4. Pulsa la tecla 'q' para salir.")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo acceder a la webcam.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if punto_clic is not None:
        cx, cy = punto_clic
        
        # Color HSV exacto del píxel donde se hizo clic
        pixel_hsv = hsv[cy, cx]
        
        # Segmentación básica del tapete verde
        mask_tapete = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        mask_objetos = cv2.bitwise_not(mask_tapete)
        
        contornos, _ = cv2.findContours(mask_objetos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        encontrado = False
        for cnt in contornos:
            if cv2.pointPolygonTest(cnt, (cx, cy), False) >= 0:
                area = cv2.contourArea(cnt)
                radio = np.sqrt(area / np.pi)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h != 0 else 0
                
                print(f"\n[OBJETO DETECTADO EN CLIC ({cx}, {cy})]")
                print(f" -> Color HSV en clic: H={pixel_hsv[0]}, S={pixel_hsv[1]}, V={pixel_hsv[2]}")
                print(f" -> Radio Equivalente: {radio:.2f} píxeles")
                print(f" -> Área: {area:.1f} px²")
                print(f" -> Bounding Box: Ancho={w}px, Alto={h}px | Aspect Ratio: {aspect_ratio:.2f}")
                encontrado = True
                break
        
        if not encontrado:
            print(f"\n[CLIC EN FONDO/TAPETE ({cx}, {cy})]")
            print(f" -> Color HSV del Tapete: H={pixel_hsv[0]}, S={pixel_hsv[1]}, V={pixel_hsv[2]}")

        punto_clic = None # Resetear evento

    cv2.imshow("CALIBRACION - Haz clic en las monedas/billetes", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()