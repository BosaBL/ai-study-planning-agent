import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

const API_BASE = "http://131.221.33.104";

export default function GuideStatus() {
  const { guideId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [guide, setGuide] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchGuide = async () => {
      try {
        const res = await axios.get(`${API_BASE}/study-guide/${guideId}`);
        setGuide(res.data);
        setLoading(false);
      } catch (err) {
        if (err.response?.status === 202) {
          setError("La guía aún se está generando. Intenta más tarde.");
        } else if (err.response?.status === 404) {
          setError("Guía no encontrada.");
        } else {
          setError("Error al obtener la guía.");
        }
        setLoading(false);
      }
    };

    fetchGuide();
  }, [guideId]);

  const generarPDF = () => {
    const doc = new jsPDF();
    const margen = 15;
    let y = 20;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.text(guide.topic, margen, y);

    y += 10;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(12);
    const resumen = doc.splitTextToSize(guide.summary, 180);
    doc.text(resumen, margen, y);
    y += resumen.length * 7 + 5;

    guide.study_plan.forEach((mod, index) => {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(13);
      doc.text(`Módulo ${index + 1}: ${mod.title}`, margen, y);
      y += 8;

      autoTable(doc, {
        startY: y,
        margin: { left: margen },
        styles: { fontSize: 10 },
        body: [
          ["Objetivos", mod.objectives.join(", ")],
          ["Temas clave", mod.key_topics.join(", ")],
          ["Actividades", mod.practical_activities.join(", ")],
          ["Recursos", mod.recommended_resources.join(", ")],
          ["Tiempo estimado", mod.estimated_time],
        ],
        theme: "grid",
        columnStyles: {
          0: { cellWidth: 45 },
          1: { cellWidth: 135 },
        },
      });

      y = doc.lastAutoTable.finalY + 10;
    });

    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.text("Fuentes Utilizadas", margen, y);
    y += 6;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    guide.sources_used.forEach((src) => {
      const lineas = doc.splitTextToSize(src, 180);
      doc.text(lineas, margen, y);
      y += lineas.length * 5;
    });

    y += 10;
    doc.setFontSize(9);
    doc.text(`Generado el: ${new Date(guide.completed_at).toLocaleString()}`, margen, y);

    doc.save(`Guia_${guide.topic}.pdf`);
  };

  if (loading) {
    return (
      <div className="text-center py-20">
        <p className="text-gray-600 text-lg animate-pulse">⏳ Cargando guía de estudio...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-20">
        <p className="text-red-600 text-lg">{error}</p>
      </div>
    );
  }

  return (
    <div className="px-4 py-12">
      <div className="max-w-4xl mx-auto bg-white rounded-xl shadow border border-gray-200 p-8">
        {/* Encabezado */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-blue-700 text-center sm:text-left break-words leading-snug mb-4">
            {guide.topic}
          </h1>
          <div className="flex flex-wrap justify-center sm:justify-start gap-3">
            <button
              onClick={() => {
                const link = `${window.location.origin}/guia/${guideId}`;
                navigator.clipboard.writeText(link);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              }}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm transition"
            >
              📋 Copiar enlace
            </button>

            <button
              onClick={generarPDF}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm transition"
            >
              📄 Descargar PDF
            </button>

            {copied && (
              <span className="text-green-600 text-sm mt-2 sm:mt-0">
                ¡Copiado!
              </span>
            )}
          </div>
        </div>

        {/* Resumen */}
        <p className="text-gray-700 text-base mb-8">{guide.summary}</p>

        {/* Plan de Estudio */}
        <h2 className="text-xl font-semibold text-blue-700 mb-4">📘 Plan de Estudio</h2>
        <div className="space-y-6">
          {guide.study_plan.map((mod, index) => (
            <div key={index}>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                Módulo {index + 1}: {mod.title}
              </h3>
              <ul className="text-sm text-gray-700 space-y-1">
                <li><strong><span className="print:hidden">🎯 </span>Objetivos:</strong> {mod.objectives.join(", ")}</li>
                <li><strong><span className="print:hidden">🔑 </span>Temas clave:</strong> {mod.key_topics.join(", ")}</li>
                <li><strong><span className="print:hidden">📝 </span>Actividades:</strong> {mod.practical_activities.join(", ")}</li>
                <li><strong><span className="print:hidden">🔗 </span>Recursos:</strong> {mod.recommended_resources.join(", ")}</li>
                <li><strong><span className="print:hidden">⏱️ </span>Tiempo estimado:</strong> {mod.estimated_time}</li>
              </ul>
            </div>
          ))}
        </div>

        {/* Fuentes */}
        <h2 className="text-xl font-semibold text-blue-700 mt-10 mb-3">🔍 Fuentes Utilizadas</h2>
        <ul className="list-disc ml-6 text-sm text-blue-700 space-y-1">
          {guide.sources_used.map((src, i) => (
            <li key={i}>
              <a
                href={src}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline"
              >
                {src}
              </a>
            </li>
          ))}
        </ul>

        {/* Fecha */}
        <p className="text-xs text-gray-500 mt-8">
          Generado el: {new Date(guide.completed_at).toLocaleString()}
        </p>

        {/* Botón volver */}
        <div className="text-center mt-10">
          <button
            onClick={() => navigate("/generar")}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-full text-base font-medium shadow transition"
          >
            🔁 Generar otra guía
          </button>
        </div>
      </div>
    </div>
  );
}
