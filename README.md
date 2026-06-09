# Sistema de Videovigilancia Multicámara

Sistema de seguridad desarrollado en Python diseñado para procesar múltiples flujos de video de forma simultánea. Integra modelos de visión computacional para la detección automática de objetos en tiempo real y gestiona el almacenamiento seguro de eventos anómalos.

## Características Principales

* **Procesamiento Concurrente:** Capacidad para manejar flujos de video simultáneos con calibración automática de cuadros por segundo (FPS).
* **Detección en Tiempo Real:** Integración del modelo YOLOv8 (`yolov8n.pt`) y OpenCV para la identificación precisa de objetos y personas.
* **Gestión de Eventos:** Funcionalidad de grabación y almacenamiento automático de clips de video ante la detección de eventos predefinidos.
* **Configuración Centralizada:** Parámetros de cámaras y umbrales de detección administrados a través de `config_multicam.json`.

## Stack Tecnológico

* Python 3.x
* OpenCV
* Ultralytics (YOLOv8)

## Instalación y Configuración

1. Clonar este repositorio:
   ```bash
   git clone [https://github.com/felipeMon20/Sistema_Vigilancia.git](https://github.com/felipeMon20/Sistema_Vigilancia.git)
