import { Link } from "react-router-dom";
import logo from "../assets/logo.png";

export default function Home() {
  return (
    <div className="max-w-2xl mx-auto p-8 bg-gray-50 rounded-xl shadow border border-gray-200 mt-10">
      <img src={logo} alt="StudyPlanner IA" className="w-36 sm:w-44 mx-auto mb-6 drop-shadow-sm" />

      <h1 className="text-3xl font-bold text-blue-700 mb-2 text-center flex items-center justify-center gap-2">
        Generación Inteligente de Guías de Estudio
      </h1>

      <p className="text-sm text-gray-500 mb-6 text-center">
        Plataforma impulsada por Inteligencia Artificial para crear planes de estudio personalizados 
        a partir de temas, palabras clave y archivos PDF. Ideal para un aprendizaje más eficiente y adaptado a ti.
      </p>

      <div className="text-center">
        <Link
          to="/generar"
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition inline-block"
        >
          🚀 Comenzar
        </Link>
      </div>
    </div>
  );
}
