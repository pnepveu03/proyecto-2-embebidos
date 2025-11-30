#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import os
import sys
import imageio
import time
import threading
from urllib.request import urlopen
from urllib.parse import urlencode

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        print("[ERROR] Instale tflite_runtime o tensorflow.")
        sys.exit(1)

# --- FUNCIÓN THINGSPEAK 
def upload_thingspeak_thread(api_key, count, top_class, top_conf):
    """
    Sube datos a ThingSpeak sin congelar el video.
    Field1: Cantidad de objetos
    Field2: Clase del objeto más probable
    Field3: Confianza de ese objeto
    """
    try:
        params = {
            'api_key': api_key,
            'field1': count,
            'field2': top_class,
            'field3': f"{top_conf:.2f}"
        }
        # Construir URL
        url = f"https://api.thingspeak.com/update?{urlencode(params)}"
        
        # Enviar petición (timeout corto para no dejar hilos colgados)
        with urlopen(url, timeout=5) as response:
            if response.status == 200:
                pass # Éxito silencioso
            else:
                print(f"[IOT] Error HTTP: {response.status}")
                
    except Exception as e:
        print(f"[IOT] Falló subida: {e}")

# --- CLASE DE ENTRADA (CÁMARA O ARCHIVO) ---
class InputWrapper:
    def __init__(self, source):
        self.mode = None
        self.cap = None
        self.reader = None
        self.iterator = None
        self.fps = 30.0 
        self.size = (640, 480)

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self.mode = "camera"
            cam_id = int(source)
            print(f"[FUENTE] Iniciando CÁMARA ID: {cam_id}...")
            self.cap = cv2.VideoCapture(cam_id)
            if not self.cap.isOpened():
                raise RuntimeError("No se pudo abrir la cámara.")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.size = (w, h)
        else:
            self.mode = "file"
            print(f"[FUENTE] Abriendo ARCHIVO: {source}")
            self.reader = imageio.get_reader(source)
            meta = self.reader.get_meta_data()
            self.fps = meta.get('fps', 24.0)
            self.size = meta.get('size', (640, 480))
            self.iterator = iter(self.reader)

    def get_frame(self):
        if self.mode == "camera":
            ret, frame_bgr = self.cap.read()
            if not ret: return False, None
            return True, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        elif self.mode == "file":
            try:
                return True, next(self.iterator)
            except StopIteration:
                return False, None

    def release(self):
        if self.mode == "camera" and self.cap:
            self.cap.release()
        elif self.mode == "file" and self.reader:
            self.reader.close()

# --- UTILS TFLITE ---
def preprocess(frame_rgb, input_size, input_dtype):
    img = cv2.resize(frame_rgb, (input_size, input_size))
    input_data = np.expand_dims(img, axis=0)
    if input_dtype == np.uint8:
        return input_data.astype(np.uint8)
    elif input_dtype == np.float32:
        return (input_data.astype(np.float32) - 127.5) / 127.5
    return input_data.astype(input_dtype)

# --- INFERENCIA EFFICIENTDET 
def run_efficientdet(interpreter, input_details, output_details, frame_rgb, conf_thres):
    input_shape = input_details[0]['shape']
    input_size = input_shape[1]
    input_dtype = input_details[0]['dtype']
    
    input_data = preprocess(frame_rgb, input_size, input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    h_img, w_img = frame_rgb.shape[:2]
    results = []

    # --- LISTA DE COSAS PERMITIDAS EN LA CALLE ---
    # Solo estos IDs serán procesados. El resto se ignoran.
    ALLOWED_CLASSES = {
        0, 1, 2, 3, 5, 6, 7,      # Vehículos y Personas
        9, 10, 11, 12, 13, 14,    # Infraestructura urbana
        15, 16, 17,               # Animales comunes (Pájaro, Gato, Perro)
        26, 27, 30                # Accesorios peatones (Mochila, Paraguas, Bolso)
    }
    # ---------------------------------------------
    
    for i in range(len(scores)):
        score = float(scores[i])
        class_id = int(classes[i])

        # Si la confianza es alta Y la clase está en nuestra lista permitida
        if score > conf_thres and class_id in ALLOWED_CLASSES:
            
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * w_img); y1 = int(ymin * h_img)
            x2 = int(xmax * w_img); y2 = int(ymax * h_img)
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w_img, x2); y2 = min(h_img, y2)
            
            results.append(([x1, y1, x2, y2], score, class_id))
            
    return results

# --- CALIBRACIÓN FPS ---
def benchmark_camera_fps(source_wrapper, interpreter, input_details, output_details):
    if source_wrapper.mode == "file":
        return source_wrapper.fps

    print("[CALIBRANDO] Midiendo velocidad real (3 seg)...")
    num_frames = 0
    start = time.time()
    while time.time() - start < 3.0:
        ret, frame = source_wrapper.get_frame()
        if not ret: break
        _ = run_efficientdet(interpreter, input_details, output_details, frame, 0.5)
        num_frames += 1
    
    real_fps = num_frames / (time.time() - start)
    safe_fps = max(1.0, real_fps * 0.9) # Margen de seguridad del 10%
    print(f"[CALIBRANDO] Detectado: {real_fps:.2f} FPS -> Ajustado a: {safe_fps:.2f} FPS")
    return safe_fps

# --- MAIN ---
def main():
    print("[INFO] Iniciando Detección + IOT ThingSpeak...")
    
    if not os.path.exists("config.yaml"):
        print("[ERROR] Falta config.yaml")
        return
    with open("config.yaml", "r") as f: cfg = yaml.safe_load(f)

    # Configs generales
    src = cfg["input"]["source"]
    output_path = cfg["output"]["path"]
    model_path = cfg["model"]["path"]
    score_thres = cfg["model"].get("score", 0.40)
    
    # Configs IOT
    ts_enabled = cfg.get("app", {}).get("thingspeak_enabled", False)
    ts_key = cfg.get("app", {}).get("thingspeak_key", "")
    ts_interval = cfg.get("app", {}).get("thingspeak_interval", 20)

    # Labels
    labels = {}
    labels_path = cfg["model"].get("labels", "classes.txt")
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = {i: line.strip() for i, line in enumerate(f.readlines())}

    # Modelo
    print(f"[INFO] Modelo: {model_path}")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Fuente
    try:
        source_wrapper = InputWrapper(src)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Calibración
    recording_fps = benchmark_camera_fps(source_wrapper, interpreter, input_details, output_details)

    # Writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path, 
        fps=recording_fps,
        codec='libx264',
        quality=None,
        macro_block_size=None,
        ffmpeg_params=['-preset', 'ultrafast'] 
    )

    frame_idx = 0
    last_upload_time = time.time()
    
    print(f"[INFO] Grabando... (Subida a nube cada {ts_interval}s: {'SI' if ts_enabled else 'NO'})")
    
    try:
        while True:
            ret, frame_rgb = source_wrapper.get_frame()
            if not ret: break

            # Detección
            detections = run_efficientdet(interpreter, input_details, output_details, frame_rgb, score_thres)
            
            # --- LÓGICA THINGSPEAK ---
            if ts_enabled and (time.time() - last_upload_time > ts_interval):
                obj_count = len(detections)
                
                # Buscar el objeto con mayor confianza (Dominante)
                top_class = 0
                top_conf = 0.0
                if obj_count > 0:
                    # Ordenar por score y tomar el primero
                    best = sorted(detections, key=lambda x: x[1], reverse=True)[0]
                    top_class = best[2] # Class ID
                    top_conf = best[1]  # Score
                
                # Iniciar HILO (Thread) para no detener el video
                print(f"[IOT] Subiendo datos: Count={obj_count}, TopClass={top_class}...")
                t = threading.Thread(
                    target=upload_thingspeak_thread, 
                    args=(ts_key, obj_count, top_class, top_conf)
                )
                t.start()
                
                last_upload_time = time.time()
            # -------------------------

            # Dibujar
            for (box, score, cls) in detections:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                lbl = labels.get(cls, str(cls))
                cv2.putText(frame_rgb, f"{lbl} {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            writer.append_data(frame_rgb)
            
            frame_idx += 1
            if frame_idx % 15 == 0:
                print(f"\rFrames: {frame_idx}", end="")

    except KeyboardInterrupt:
        print("\n[INFO] Detenido.")
    finally:
        source_wrapper.release()
        writer.close()
        print("\n[INFO] Finalizado.")

if __name__ == "__main__":
    main()
