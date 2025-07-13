import { useParams, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";

const API_BASE = "http://131.221.33.104";

export default function Generando() {
  const { guideId } = useParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/study-guide/${guideId}`);

        if (res.status === 200) {
          clearInterval(interval);
          navigate(`/guia/${guideId}`);
        } else if (res.status === 202) {
          setStatus("pending");
        }
      } catch (err) {
        if (err.response?.status === 404) {
          setStatus("not_found");
          setError("❌ Guía no encontrada.");
          clearInterval(interval);
        } else if (err.response?.status === 202) {
          setStatus("pending");
        } else {
          setStatus("error");
          setError("❌ Error al cargar la guía. Intenta nuevamente.");
          clearInterval(interval);
        }
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [guideId, navigate]);

  return (
    <div className="px-4 mt-10 flex justify-center">
      <div className="max-w-xl w-full bg-gray-50 p-8 rounded-xl shadow border border-gray-200 text-center">
        {status === "loading" || status === "pending" ? (
          <>
            <div className="w-12 h-12 border-4 border-blue-400 border-dashed rounded-full animate-spin mx-auto mb-6" />
            <h1 className="text-2xl font-bold text-blue-700 mb-2 flex items-center justify-center gap-2">
              <span>🔄</span> Generando Guía de Estudio
            </h1>
            <p className="text-sm text-gray-600 mb-2">
              El sistema está procesando tu información para crear una guía personalizada.
            </p>
            <p className="text-sm text-gray-400">
              No cierres esta ventana. Serás redirigido automáticamente cuando esté lista.
            </p>
          </>
        ) : (
          <>
            <h2 className="text-xl font-semibold text-red-600 mb-2">Ocurrió un problema</h2>
            <p className="text-sm text-gray-600">{error}</p>
          </>
        )}
      </div>
    </div>
  );
}
