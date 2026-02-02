import threading
import cv2
import time
import os
import json
import math 
import collections
import queue # NUEVO: Para la cola de grabación
from datetime import datetime
from PIL import Image, ImageTk
import customtkinter as ctk
from ultralytics import YOLO

# =============================================================================
# GESTOR DE CONFIGURACIÓN (AHORA CON AJUSTES GLOBALES)
# =============================================================================
class ConfigManager:
    """
    Gestiona cámaras y ahora también AJUSTES GLOBALES.
    """
    def __init__(self, archivo="config_multicam.json"):
        self.archivo = archivo
        # Estructura base por defecto
        self.datos = {
            "ajustes": {
                "cooldown_segundos": 5,
                "calibracion_segundos": 3,
                "conf_persona": 0.60,
                "conf_animal": 0.35,
                "conf_vehiculo": 0.50
            },
            "camaras_guardadas": []
        }
        self.cargar()

    def cargar(self):
        if os.path.exists(self.archivo):
            try:
                with open(self.archivo, "r") as f:
                    cargado = json.load(f)
                    # Fusionar con defaults para evitar errores si faltan claves
                    self.datos.update(cargado)
                    if "ajustes" not in self.datos: # Parche para versiones viejas
                        self.datos["ajustes"] = {
                            "cooldown_segundos": 5, "calibracion_segundos": 3,
                            "conf_persona": 0.60, "conf_animal": 0.35, "conf_vehiculo": 0.50
                        }
            except Exception as e:
                print(f"[ERROR CONFIG] {e}")

    def guardar(self):
        try:
            with open(self.archivo, "w") as f:
                json.dump(self.datos, f, indent=4)
        except Exception as e:
            print(f"[ERROR GUARDAR] {e}")

    def get_ajuste(self, clave):
        return self.datos["ajustes"].get(clave, 0)

    # --- Gestión de Cámaras ---
    def agregar_config(self, nombre, origen):
        for cam in self.datos["camaras_guardadas"]:
            if cam["nombre"] == nombre and cam["origen"] == origen: return
        self.datos["camaras_guardadas"].append({"nombre": nombre, "origen": origen, "activa": True})
        self.guardar()

    def borrar_config(self, nombre, origen):
        self.datos["camaras_guardadas"] = [c for c in self.datos["camaras_guardadas"] if not (c["nombre"] == nombre and c["origen"] == origen)]
        self.guardar()

    def obtener_guardadas(self):
        return self.datos.get("camaras_guardadas", [])


# =============================================================================
# VISTA DE CÁMARA (ARQUITECTURA MULTI-HILO PRO)
# =============================================================================
class VistaCamara(ctk.CTkFrame):
    def __init__(self, parent, nombre, origen, model, ruta_grabaciones, on_delete_callback, on_click_callback, config):
        super().__init__(parent)
        self.nombre = nombre
        self.origen = origen
        self.model = model
        self.ruta_grabaciones = ruta_grabaciones
        self.on_delete = on_delete_callback
        self.on_click = on_click_callback
        self.config = config # Acceso a los ajustes
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.lbl_video = ctk.CTkLabel(self, text=f"Iniciando {nombre}...", fg_color="#101010", corner_radius=0)
        self.lbl_video.grid(row=0, column=0, sticky="nsew")
        self.lbl_video.bind("<Button-1>", self.al_hacer_clic)
        
        self.btn_cerrar = ctk.CTkButton(self, text="X", width=20, height=20, fg_color="#922B21", hover_color="#C0392B", command=self.solicitar_cierre)
        self.btn_cerrar.place(relx=0.98, rely=0.02, anchor="ne")

        # Variables de estado
        self.running = True
        self.grabando = False
        self.ultimo_evento = 0
        self.frame_actual = None 
        self.lock = threading.Lock()
        self.estado_conexion = "CONECTANDO"

        # Leer configuración
        self.cooldown = self.config.get_ajuste("cooldown_segundos")
        self.tiempo_calibracion = self.config.get_ajuste("calibracion_segundos")

        # Técnico
        self.fps_real = 20.0 
        self.buffer_ram = collections.deque(maxlen=60)
        self.inicio_grabacion_ts = 0
        self.objetos_detectados_evento = set()

        # --- SISTEMA DE GRABACIÓN ASÍNCRONA (NUEVO) ---
        self.cola_grabacion = queue.Queue()
        self.hilo_escritor = threading.Thread(target=self.loop_escritura_disco, daemon=True)
        self.hilo_escritor.start()

        # Hilo de procesamiento principal
        self.thread = threading.Thread(target=self.loop_procesamiento)
        self.thread.start()
        
        self.loop_visual()

    def al_hacer_clic(self, event):
        if self.on_click: self.on_click(self)

    # -------------------------------------------------------------------------
    # HILO 1: CAPTURA Y DETECCIÓN (Rápido)
    # -------------------------------------------------------------------------
    def loop_procesamiento(self):
        if isinstance(self.origen, int): cap = cv2.VideoCapture(self.origen, cv2.CAP_DSHOW)
        else: cap = cv2.VideoCapture(self.origen)

        if not cap.isOpened():
            self.estado_conexion = "ERROR"
            return

        t_inicio = time.time()
        frames = 0
        calibrado = False

        # Umbrales desde config
        CONF_HUMANO = self.config.get_ajuste("conf_persona")
        CONF_ANIMAL = self.config.get_ajuste("conf_animal")
        CONF_VEHICULO = self.config.get_ajuste("conf_vehiculo")

        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    self.estado_conexion = "ERROR"
                    time.sleep(1.0)
                    continue
                
                self.estado_conexion = "OK"
                t_actual = time.time()
                
                # --- CALIBRACIÓN ---
                if (t_actual - t_inicio) < self.tiempo_calibracion:
                    frames += 1
                    frame_anotado = frame.copy()
                    fps = frames / ((t_actual - t_inicio) + 0.001)
                    cv2.putText(frame_anotado, f"CALIBRANDO... ({fps:.1f} FPS)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    if not calibrado:
                        self.fps_real = max(5, min(frames / self.tiempo_calibracion, 60))
                        # Buffer ajustado a FPS reales
                        self.buffer_ram = collections.deque(maxlen=int(self.fps_real * 3))
                        calibrado = True

                    frame_anotado = frame.copy()
                    detectado = False
                    nombres = []

                    # --- IA ---
                    try:
                        results = self.model(frame, stream=True, classes=[0,2,3,15,16], conf=0.25, verbose=False)
                        for r in results:
                            for box in r.boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                label_cls = self.model.names[cls]
                                
                                # Filtros dinámicos
                                dibujar = False
                                if cls == 0 and conf >= CONF_HUMANO: dibujar = True
                                elif cls in [15, 16] and conf >= CONF_ANIMAL: dibujar = True 
                                elif cls in [2, 3] and conf >= CONF_VEHICULO: dibujar = True

                                if dibujar:
                                    detectado = True
                                    nombres.append(label_cls)
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    color = (255, 0, 0) if cls == 0 else (0, 255, 0)
                                    cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color, 2)
                                    # Etiqueta
                                    lbl = f"{label_cls} {conf:.2f}"
                                    (w, h), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    cv2.rectangle(frame_anotado, (x1, y1-20), (x1+w, y1), color, -1)
                                    cv2.putText(frame_anotado, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    except: pass

                    # --- LÓGICA DE GRABACIÓN ---
                    self.buffer_ram.append(frame_anotado)

                    if detectado:
                        self.ultimo_evento = t_actual
                        self.objetos_detectados_evento.update(nombres)
                        if not self.grabando:
                            h, w = frame.shape[:2]
                            self.iniciar_grabacion(w, h)

                    if self.grabando:
                        # EN LUGAR DE ESCRIBIR, ENVIAMOS A LA COLA
                        self.cola_grabacion.put(("WRITE", frame_anotado))
                        
                        if (t_actual - self.ultimo_evento) > self.cooldown:
                            self.detener_grabacion()

                # Overlay UI
                cv2.rectangle(frame_anotado, (0,0), (len(self.nombre)*15+20, 30), (0,0,0), -1) 
                cv2.putText(frame_anotado, self.nombre.upper(), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if self.grabando:
                     cv2.circle(frame_anotado, (len(self.nombre)*15+40, 15), 6, (0, 0, 255), -1) 
                     cv2.putText(frame_anotado, "REC", (len(self.nombre)*15+55, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                with self.lock:
                    self.frame_actual = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)

            except cv2.error:
                self.estado_conexion = "ERROR"
                time.sleep(2)
            except Exception:
                time.sleep(0.1)

        if cap.isOpened(): cap.release()
        # Avisar al hilo escritor que termine
        self.cola_grabacion.put(("EXIT", None))

    # -------------------------------------------------------------------------
    # HILO 2: ESCRITURA EN DISCO (Lento, no bloquea)
    # -------------------------------------------------------------------------
    def loop_escritura_disco(self):
        writer = None
        
        while True:
            # Esperar mensajes de la cola
            try:
                comando, payload = self.cola_grabacion.get(timeout=2)
            except queue.Empty:
                continue

            if comando == "START":
                # Payload: (path, fps, w, h)
                path, fps, w, h = payload
                if writer is not None: writer.release()
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            elif comando == "WRITE":
                # Payload: frame
                if writer is not None:
                    writer.write(payload)
            
            elif comando == "STOP":
                if writer is not None:
                    writer.release()
                    writer = None
            
            elif comando == "EXIT":
                if writer is not None: writer.release()
                break

    # --- CONTROLADORES ---
    def iniciar_grabacion(self, w, h):
        # Nombre con milisegundos (%f) para evitar colisiones
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.base_name = os.path.join(self.ruta_grabaciones, f"{self.nombre}_{ts}")
        video_path = self.base_name + ".mp4"
        
        print(f"[REC] Nueva grabación: {video_path}")
        
        # 1. Enviar comando START al hilo escritor
        self.cola_grabacion.put(("START", (video_path, self.fps_real, w, h)))
        
        # 2. Volcar el buffer (pasado) a la cola de escritura
        for f in self.buffer_ram:
            self.cola_grabacion.put(("WRITE", f))
            
        self.grabando = True
        self.inicio_grabacion_ts = time.time()
        self.objetos_detectados_evento = set()

    def detener_grabacion(self):
        # Enviar comando STOP
        self.cola_grabacion.put(("STOP", None))
        self.grabando = False
        
        # Generar JSON (Metadata) en el hilo principal (es rápido)
        try:
            info = {
                "camara": self.nombre,
                "inicio": datetime.fromtimestamp(self.inicio_grabacion_ts).strftime("%Y-%m-%d %H:%M:%S"),
                "duracion_seg": round(time.time() - self.inicio_grabacion_ts, 2),
                "fps": round(self.fps_real, 2),
                "detecciones": list(self.objetos_detectados_evento)
            }
            with open(self.base_name + ".json", "w") as f:
                json.dump(info, f, indent=4)
        except: pass

    # --- UI ---
    def loop_visual(self):
        if not self.running: return
        if self.estado_conexion != "OK":
            txt = "[ERROR]" if self.estado_conexion == "ERROR" else "Cargando..."
            self.lbl_video.configure(text=f"{self.nombre}\n{txt}", image=None)
        else:
            img_pil = None
            with self.lock:
                if self.frame_actual is not None: img_pil = Image.fromarray(self.frame_actual)
            
            if img_pil:
                w_w = self.lbl_video.winfo_width()
                h_w = self.lbl_video.winfo_height()
                if w_w > 10 and h_w > 10:
                    r = img_pil.width / img_pil.height
                    nw, nh = w_w, int(w_w / r)
                    if nh > h_w: nh, nw = h_w, int(h_w * r)
                    if nw > 10 and nh > 10:
                        ctk_img = ctk.CTkImage(light_image=img_pil, size=(nw, nh))
                        self.lbl_video.configure(image=ctk_img, text="")
        self.after(30, self.loop_visual)

    def solicitar_cierre(self): self.on_delete(self)
    def detener(self):
        self.running = False
        self.grid_forget()
        self.destroy()

# =============================================================================
# APP PRINCIPAL
# =============================================================================
class AppSeguridad(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.directorio = os.path.dirname(os.path.abspath(__file__))
        self.ruta_videos = os.path.join(self.directorio, "grabaciones")
        if not os.path.exists(self.ruta_videos): os.makedirs(self.ruta_videos)
        
        self.config = ConfigManager(os.path.join(self.directorio, "config_multicam.json"))
        print("[SISTEMA] Cargando IA...")
        self.model = YOLO("yolov8n.pt")
        
        self.title("Sistema V9.0 - Async Recorder & Settings")
        self.geometry("1300x850")
        ctk.set_appearance_mode("Dark")
        
        self.grid_columnconfigure(0, weight=0) 
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar_visible = True
        self.camaras_activas_widgets = [] 
        self.camara_en_foco = None

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="PANEL DE CONTROL", font=("Arial", 16, "bold")).pack(pady=(15, 10))
        
        self.frm_add = ctk.CTkFrame(self.sidebar)
        self.frm_add.pack(padx=10, pady=5, fill="x")
        self.ent_nombre = ctk.CTkEntry(self.frm_add, placeholder_text="Nombre")
        self.ent_nombre.pack(fill="x", padx=5, pady=5)
        self.combo_origen = ctk.CTkComboBox(self.frm_add, values=["Escaneando..."])
        self.combo_origen.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(self.frm_add, text="AGREGAR", command=self.agregar_nueva, fg_color="#2E86C1").pack(pady=5, fill="x")
        
        ctk.CTkLabel(self.sidebar, text="Cámaras Guardadas:", font=("Arial", 12)).pack(pady=(20, 5))
        self.lista_camaras = ctk.CTkScrollableFrame(self.sidebar)
        self.lista_camaras.pack(padx=10, fill="both", expand=True)
        ctk.CTkButton(self.sidebar, text="Escanear Puertos USB", command=self.escanear_puertos, fg_color="#555").pack(pady=10, padx=10, fill="x")
        
        # Main
        self.main_container = ctk.CTkFrame(self, corner_radius=0, fg_color="#000000")
        self.main_container.grid(row=0, column=1, sticky="nsew")
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        self.header = ctk.CTkFrame(self.main_container, height=40, corner_radius=0, fg_color="#202020")
        self.header.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(self.header, text="Menu", width=50, fg_color="transparent", font=("Arial", 12, "bold"), hover_color="#404040", command=self.toggle_sidebar).pack(side="left", padx=5)
        self.lbl_modo = ctk.CTkLabel(self.header, text="Vista: Mosaico", font=("Arial", 12, "bold"))
        self.lbl_modo.pack(side="left", padx=10)
        
        self.area_video = ctk.CTkFrame(self.main_container, fg_color="#000000", corner_radius=0)
        self.area_video.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        self.after(100, self.escanear_puertos)
        self.after(500, self.cargar_configuracion_inicial)

    # --- LÓGICA UI ---
    def actualizar_layout(self):
        for i in range(10): 
             self.area_video.grid_columnconfigure(i, weight=0)
             self.area_video.grid_rowconfigure(i, weight=0)
        total = len(self.camaras_activas_widgets)
        if total == 0: return

        if self.camara_en_foco is None or total == 1:
            self.lbl_modo.configure(text="Vista: Mosaico")
            self.camara_en_foco = None 
            cols = math.ceil(math.sqrt(total))
            rows = math.ceil(total / cols)
            for c in range(cols): self.area_video.grid_columnconfigure(c, weight=1)
            for r in range(rows): self.area_video.grid_rowconfigure(r, weight=1)
            for i, cam in enumerate(self.camaras_activas_widgets):
                r = i // cols
                c = i % cols
                cam.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
                cam.tkraise()
        else:
            self.lbl_modo.configure(text=f"Vista: Foco en {self.camara_en_foco.nombre}")
            self.area_video.grid_columnconfigure(0, weight=6)
            self.area_video.grid_columnconfigure(1, weight=1)
            self.area_video.grid_rowconfigure(0, weight=1)
            self.camara_en_foco.grid(row=0, column=0, rowspan=10, sticky="nsew", padx=2, pady=2)
            self.camara_en_foco.tkraise()
            otras = [c for c in self.camaras_activas_widgets if c != self.camara_en_foco]
            for i in range(len(otras)): self.area_video.grid_rowconfigure(i, weight=1)
            for i, cam in enumerate(otras):
                cam.grid(row=i, column=1, sticky="nsew", padx=2, pady=2)
                cam.tkraise()

    def al_hacer_clic_en_camara(self, camara_obj):
        if self.camara_en_foco == camara_obj: self.camara_en_foco = None
        else: self.camara_en_foco = camara_obj
        self.actualizar_layout()

    def toggle_sidebar(self):
        if self.sidebar_visible: self.sidebar.grid_forget(); self.sidebar_visible = False
        else: self.sidebar.grid(row=0, column=0, sticky="nsew"); self.sidebar_visible = True

    def escanear_puertos(self):
        puertos = []
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: puertos.append(f"{i} - Disp {i}")
                cap.release()
        self.combo_origen.configure(values=puertos if puertos else ["Ninguno"])
        if puertos: self.combo_origen.set(puertos[0])

    def cargar_configuracion_inicial(self):
        for widget in self.lista_camaras.winfo_children(): widget.destroy()
        for cam_data in self.config.obtener_guardadas(): self.crear_item_lista(cam_data["nombre"], cam_data["origen"])

    def crear_item_lista(self, nombre, origen):
        row = ctk.CTkFrame(self.lista_camaras, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=nombre, anchor="w").pack(side="left", padx=5)
        ctk.CTkButton(row, text="X", width=30, fg_color="#922B21", command=lambda: self.eliminar_config(nombre, origen)).pack(side="right", padx=2)
        ctk.CTkButton(row, text="Ver", width=40, fg_color="#2E86C1", command=lambda: self.toggle_camara(nombre, origen)).pack(side="right", padx=2)

    def agregar_nueva(self):
        nombre, sel = self.ent_nombre.get(), self.combo_origen.get()
        if not nombre: return
        try: origen = int(sel.split(" - ")[0])
        except: origen = sel
        self.config.agregar_config(nombre, origen)
        self.crear_item_lista(nombre, origen)
        self.activar_camara(nombre, origen)

    def eliminar_config(self, n, o):
        self.config.borrar_config(n, o)
        self.cerrar_vista_camara(n, o)
        self.cargar_configuracion_inicial()

    def toggle_camara(self, n, o):
        for cam in self.camaras_activas_widgets:
            if cam.nombre == n and cam.origen == o: self.cerrar_vista_camara(n, o); return
        self.activar_camara(n, o)

    def activar_camara(self, nombre, origen):
        for cam in self.camaras_activas_widgets:
            if cam.nombre == nombre and cam.origen == origen: return
        if isinstance(origen, int):
            for cam in self.camaras_activas_widgets:
                if cam.origen == origen: print(f"Dispositivo {origen} ocupado"); return

        # Pasamos self.config a la cámara para que lea los ajustes
        vista = VistaCamara(self.area_video, nombre, origen, self.model, self.ruta_videos, 
                            self.callback_cierre_vista, self.al_hacer_clic_en_camara, self.config)
        self.camaras_activas_widgets.append(vista)
        self.actualizar_layout()

    def cerrar_vista_camara(self, nombre, origen):
        for cam in self.camaras_activas_widgets:
            if cam.nombre == nombre and cam.origen == origen:
                cam.detener()
                self.camaras_activas_widgets.remove(cam)
                if self.camara_en_foco == cam: self.camara_en_foco = None
                break
        self.actualizar_layout()

    def callback_cierre_vista(self, v): self.cerrar_vista_camara(v.nombre, v.origen)
    def cerrar_app(self):
        for cam in self.camaras_activas_widgets: cam.running = False
        self.destroy()

if __name__ == "__main__":
    app = AppSeguridad()
    app.protocol("WM_DELETE_WINDOW", app.cerrar_app)
    app.mainloop()