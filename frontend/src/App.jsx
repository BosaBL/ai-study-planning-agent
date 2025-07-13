import { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import GenerateGuide from "./pages/GenerateGuide";
import HowItWorks from "./pages/HowItWorks";
import GuideStatus from "./pages/GuideStatus";
import Home from "./pages/Home";
import Generando from "./pages/Generando";
import logo from "./assets/logo3.png";

function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <Router>
      <div className="flex flex-col min-h-screen font-sans">
        {/* Header fijo con sombra */}
        <header className="fixed top-0 left-0 w-full bg-white shadow-md px-4 py-3 flex justify-between items-center z-50">
          <Link to="/" className="flex items-center">
            <img src={logo} alt="Logo StudyPlanner" className="w-28 object-contain" />
          </Link>

          {/* Botón menú hamburguesa */}
          <button onClick={() => setMenuOpen(!menuOpen)} className="sm:hidden">
            <svg className="w-6 h-6 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          {/* Menú en pantallas grandes */}
          <nav className="hidden sm:flex gap-6 text-gray-700 font-medium text-sm">
            <Link to="/" className="hover:text-blue-700 transition">Inicio</Link>
            <Link to="/generar" className="hover:text-blue-700 transition">Generar Guía</Link>
            <Link to="/como-funciona" className="hover:text-blue-700 transition">¿Cómo funciona?</Link>
          </nav>
        </header>

        {/* Menú móvil */}
        {menuOpen && (
          <nav className="sm:hidden bg-white px-4 py-3 shadow-md mt-[72px] z-40">
            <ul className="space-y-2 text-gray-700">
              <li><Link to="/" onClick={() => setMenuOpen(false)}>Inicio</Link></li>
              <li><Link to="/generar" onClick={() => setMenuOpen(false)}>Generar Guía</Link></li>
              <li><Link to="/como-funciona" onClick={() => setMenuOpen(false)}>¿Cómo funciona?</Link></li>
            </ul>
          </nav>
        )}

        {/* Contenido principal con espacio arriba */}
        <main className="flex-grow bg-gray-50 pt-24 px-4 pb-20">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/generar" element={<GenerateGuide />} />
            <Route path="/como-funciona" element={<HowItWorks />} />
            <Route path="/guia/:guideId" element={<GuideStatus />} />
            <Route path="/generando/:guideId" element={<Generando />} />
          </Routes>
        </main>

        {/* Footer profesional */}
        <footer className="bg-gray-100 shadow-inner text-gray-600 text-sm py-6 px-4">
          <div className="max-w-5xl mx-auto flex flex-col items-center text-center space-y-2">
            <p className="text-base font-semibold text-gray-800">
              📘 Proyecto Final — Asistente de Estudio con IA
            </p>
            <p>Ingeniería Civil en Informática · Universidad de Los Lagos</p>
            <p>
              Desarrollado por <strong>Christopher Sepúlveda</strong>, <strong>Constanza Jaramillo</strong> y <strong>Carolina Hernández</strong>
            </p>

            {/* Enlaces a GitHub y otros recursos */}
            <div className="flex gap-4 mt-2">
              <a
                href="https://github.com/BosaBL/ai-study-planning-agent"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-blue-600 transition underline"
              >
                Repositorio del Proyecto
              </a>
            </div>

            <p className="text-xs text-gray-400">© {new Date().getFullYear()} Todos los derechos reservados.</p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;

