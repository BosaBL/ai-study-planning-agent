# 🧠 AI Study Assistant – Frontend

Este es el frontend del proyecto **AI Study Assistant**, una plataforma que permite generar guías de estudio personalizadas a partir de documentos PDF utilizando inteligencia artificial. Está desarrollado en **React** con **Vite**, **Tailwind CSS**, **React Router**, y otras librerías modernas como `react-query` y `axios`.

## 🌐 Características

- Carga de archivos PDF para análisis.
- Generación asíncrona de guías de estudio.
- Vista detallada de guías con plan de estudio y fuentes.
- Descarga de guía en PDF.
- Copia rápida de enlace.
- Diseño responsivo y moderno.

---

## 🚀 Requisitos

- Node.js ≥ 18
- npm ≥ 9

---

## ⚙️ Instalación y Ejecución

1. **Clona este repositorio** o copia el frontend a tu máquina:

```bash
git clone https://github.com/BosaBL/ai-study-planning-agent.git
cd frontend-ai-study-assistant
```

2. Instala las dependencias:

```bash
npm install
```

3. Configura las variables si es necesario
(Actualmente se conecta a http://131.221.33.104 desde el archivo fuente).

4. Ejecuta el servidor de desarrollo:

```bash
npm run dev
```

5. Abre tu navegador y visita:
http://localhost:5173

---

## 📁 Estructura Principal

```bash
.
├── public/                   # Archivos estáticos
├── src/
│   ├── routes/              # Rutas de la app
│   ├── components/          # Componentes reutilizables
│   ├── pages/               # Páginas principales
│   ├── styles/              # Estilos personalizados (si hay)
│   ├── App.tsx              # Componente raíz
│   └── main.tsx             # Entrada de React
├── tailwind.config.js
├── vite.config.ts
└── package.json
```

---

## 📦 Tecnologías Usadas

- React 18
- Vite
- Tailwind CSS
- React Router DOM
- React Query
- Axios
- jsPDF + AutoTable