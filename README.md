<div align="center">
  <img src="https://github.com/BosaBL/ai-study-planning-agent/blob/main/frontend/public/logo.png" alt="Logo del Proyecto" width="150"/>
  <h1>Study Agent: Tu Tutor Personalizado con IA</h1>
  <p>
    <strong>Transforma cualquier documento o tema en una guía de estudio completa y estructurada.</strong>
  </p>
  <p>
    <a href="#%EF%B8%8F-tecnologías-utilizadas">Tecnologías</a> •
    <a href="#brain-cómo-funciona-el-agente-inteligente">Cómo Funciona</a> •
    <a href="#-instalación-y-ejecución">Instalación</a> •
    <a href="#-estructura-del-proyecto">Estructura</a>
  </p>
</div>

---

## :rocket: ¿Qué es Study Agent?

**Study Agent** es una aplicación web inteligente que actúa como un tutor personal. Simplemente proporciona un tema, sube tus documentos PDF (apuntes de clase, libros, artículos) y el agente de IA investigará, analizará y creará una guía de estudio a medida para ti.

El sistema no solo extrae información, sino que la enriquece con búsquedas web para ofrecer una perspectiva más completa y actualizada, construyendo una base de conocimientos que se vuelve más experta con cada uso.

## :brain: ¿Cómo Funciona? El Agente Inteligente

El corazón de este proyecto es un **agente autónomo de IA** construido en Python. Este agente utiliza un avanzado pipeline de **Generación Aumentada por Recuperación (RAG)** para garantizar que el contenido sea preciso, relevante y basado en fuentes verificables.

No se limita a responder preguntas; construye activamente una base de conocimientos vectorial (usando ChromaDB) a partir de los documentos que le proporcionas y las búsquedas que realiza.

> :point_right: **Para una explicación técnica detallada sobre la arquitectura del agente, su flujo de datos y cómo se vuelve más inteligente con el tiempo, consulta el [README del Backend](https://github.com/BosaBL/ai-study-planning-agent/blob/main/backend/README.md).**

## ✨ Características Principales

- **Frontend Moderno y Reactivo:** Una interfaz de usuario limpia y fácil de usar construida con React y Tailwind CSS.
- **Generación Asíncrona:** Pide tu guía y sigue usando la aplicación. Se te notificará cuando esté lista.
- **Carga de PDFs:** Enriquece la base de conocimientos del agente con tus propios documentos.
- **Planes de Estudio Detallados:** Recibe un resumen completo, un plan de estudio paso a paso y las fuentes utilizadas.
- **Base de Conocimientos Persistente:** El agente "recuerda" y aprende de interacciones pasadas.

## 🛠️ Tecnologías Utilizadas

| Área                    | Tecnología                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| :---------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| :computer: **Frontend** | ![React](https://img.shields.io/badge/React-19-blue?style=for-the-badge&logo=react) ![Vite](https://img.shields.io/badge/Vite-6.3-purple?style=for-the-badge&logo=vite) ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-blue?style=for-the-badge&logo=tailwindcss)                                                                                                                                                                                                                                   |
| :robot: **Backend**     | ![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=for-the-badge&logo=fastapi) ![LangChain](https://img.shields.io/badge/LangChain-0.2-purple?style=for-the-badge&logo=langchain) ![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-orange?style=for-the-badge&logo=google-gemini) ![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-blueviolet?style=for-the-badge&logo=chromadb) |

## 🚀 Instalación y Ejecución

Este es un proyecto monorepo con dos partes principales: `frontend` y `backend`. Deberás instalar y ejecutar cada una por separado.

### Backend (El Agente IA)

1. **Navega a la carpeta del backend:**

   ```bash
   cd backend
   ```

2. **Instala las dependencias:**
   (Se recomienda usar `uv`, el gestor de paquetes de Python usado en el proyecto)

   ```bash
   uv sync
   ```

3. **Configura tus credenciales:**
   Crea un archivo `.env` en la carpeta `backend` y a��ade tus claves de API. Consulta la [guía de configuración del backend](https://github.com/BosaBL/study-agent/blob/main/backend/README.md#-instalación-y-configuración) para más detalles.

4. **Ejecuta el servidor:**

   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   El backend estará disponible en `http://localhost:8000`.

### Frontend (La Interfaz de Usuario)

1. **Abre una nueva terminal y navega a la carpeta del frontend:**

   ```bash
   cd frontend
   ```

2. **Instala las dependencias:**

   ```bash
   npm install
   ```

3. **Ejecuta el servidor de desarrollo:**

   ```bash
   npm run dev
   ```

   La aplicación web estará disponible en `http://localhost:5173` (o el puerto que indique Vite).

## 📂 Estructura del Proyecto

El repositorio está organizado como un monorepo para separar claramente las responsabilidades:

```
/
├── 📁 backend/         # Todo el código del agente IA (Python, FastAPI, LangChain)
│   ├── main.py         # Punto de entrada de la API
│   ├── README.md       # Documentación técnica detallada del backend
│   └── ...
├── 📁 frontend/        # Todo el código de la interfaz de usuario (React, Vite)
│   ├── src/            # Código fuente de la aplicación React
│   ├── README.md       # Documentación específica del frontend
│   └── ...
└── 📄 README.md         # Este archivo: la vista general del proyecto.
```
