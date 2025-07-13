import { useForm } from "react-hook-form";
import { useState } from "react";
import { uploadPDFs, generateGuide } from "../api/guideService";
import { useNavigate } from "react-router-dom";

const MAX_FILES = 10;
const MAX_KEYWORDS = 3;

export default function GenerateGuide() {
  const navigate = useNavigate();
  const { register, handleSubmit, setError, formState: { errors } } = useForm();
  const [pdfs, setPdfs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const onSubmit = async (data) => {
    const keywords = data.search_queries.split(",").map(k => k.trim()).filter(k => k !== "");
    if (keywords.length > MAX_KEYWORDS) {
      setError("search_queries", { message: "Máximo 3 palabras clave separadas por coma." });
      return;
    }

    if (pdfs.length === 0) {
      setMessage("⚠️ Para obtener una guía de estudio de calidad, se recomienda subir contenido en formato PDF.");
    } else {
      setMessage("");
    }

    setLoading(true);

    try {
      if (pdfs.length > 0) {
        await uploadPDFs(pdfs);
      }

      const response = await generateGuide({
        topic: data.topic,
        search_queries: keywords,
        depth: data.depth,
      });

      const guideId = response.data.guide_id;
      navigate(`/generando/${guideId}`);
    } catch (error) {
      setMessage("❌ Error al generar la guía.");
      console.error(error);
    }

    setLoading(false);
  };

  return (
    <div className="max-w-2xl mx-auto p-8 bg-gray-50 rounded-xl shadow border border-gray-200 mt-10">
      {/* Encabezado bonito */}
      <h1 className="text-3xl font-bold text-blue-700 mb-2 text-center flex items-center justify-center gap-2">
        Generar Guía de Estudio
      </h1>
      <p className="text-center text-gray-500 mb-6 text-sm">
        Personaliza tu plan de estudio con IA usando tus documentos PDF.
      </p>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* Tema */}
        <div>
          <label className="block text-gray-700 font-medium mb-1">
            ¿Qué quieres aprender? <span className="text-red-500">*</span>
          </label>
          <input
            {...register("topic", { required: "Campo obligatorio" })}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="Ej: Quiero aprender cómo se aplica la IA en medicina para diagnóstico"
          />
          <p className="text-sm text-gray-500 mt-1">
            Sé específica/o. Ejemplo: “Quiero entender cómo se usa la IA en el diagnóstico médico” o “Quiero aprender sobre visión por computadora para drones”.
          </p>
          {errors.topic && <p className="text-red-600 mt-1">{errors.topic.message}</p>}
        </div>

        {/* Nivel */}
        <div>
          <label className="block text-gray-700 font-medium mb-1">
            Nivel <span className="text-red-500">*</span>
          </label>
          <select
            {...register("depth")}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            <option value="básico">Básico</option>
            <option value="intermedio">Intermedio</option>
            <option value="avanzado">Avanzado</option>
          </select>
        </div>

        {/* Palabras clave */}
        <div>
          <label className="block text-gray-700 font-medium mb-1">
            Palabras clave (separadas por coma)
          </label>
          <input
            {...register("search_queries")}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="Ej: redes neuronales, aprendizaje profundo"
          />
          <p className="text-sm text-gray-500 mt-1">Opcional. Máximo 3 palabras clave.</p>
          {errors.search_queries && <p className="text-red-600 mt-1">{errors.search_queries.message}</p>}
        </div>

        <hr className="my-4 border-gray-300" />

        {/* Archivos PDF */}
        <div>
          <label className="block text-gray-700 font-medium mb-1">Sube tus archivos PDF</label>
          <input
            type="file"
            accept=".pdf"
            multiple
            disabled={pdfs.length >= MAX_FILES}
            className="block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:border-0 file:rounded-lg file:bg-blue-100 file:text-blue-700 hover:file:bg-blue-200 disabled:opacity-50"
            onChange={(e) => {
              const newFiles = Array.from(e.target.files);
              const updatedFiles = [...pdfs];

              for (const file of newFiles) {
                if (!updatedFiles.some((f) => f.name === file.name && f.size === file.size)) {
                  updatedFiles.push(file);
                }
              }

              if (updatedFiles.length > MAX_FILES) {
                setMessage("❌ Límite alcanzado: solo puedes subir hasta 10 archivos PDF.");
                return;
              }

              // ✅ Validar que el total no supere 400 MB
              const totalSize = updatedFiles.reduce((acc, file) => acc + file.size, 0);
              const maxTotalSize = 400 * 1024 * 1024; // 400MB en bytes

              if (totalSize > maxTotalSize) {
                setMessage("❌ El tamaño total de los archivos excede el límite de 400MB.");
                return;
              }

              setMessage("");
              setPdfs(updatedFiles);
            }}
          />

          {pdfs.length > 0 && (
            <ul className="mt-3 space-y-2 text-sm text-gray-700">
              {pdfs.map((file, index) => (
                <li key={index} className="flex items-center justify-between bg-gray-100 px-3 py-2 rounded shadow-sm">
                  <span className="truncate">{file.name}</span>
                  <button
                    type="button"
                    onClick={() => setPdfs(pdfs.filter((_, i) => i !== index))}
                    className="text-red-500 hover:underline text-xs"
                  >
                    Quitar
                  </button>
                </li>
              ))}
            </ul>
          )}

          <p className="text-sm text-gray-500 mt-1">
            Opcional. Máximo 10 archivos en formato <strong>.pdf</strong> y un total de hasta <strong>400MB</strong>.
          </p>

          {pdfs.length >= MAX_FILES && (
            <p className="text-sm text-red-600 mt-2">
              ⚠️ Has alcanzado el límite de 10 archivos PDF.
            </p>
          )}
        </div>

        {/* Alerta si no hay PDFs */}
        {pdfs.length === 0 && (
          <div className="bg-yellow-100 text-yellow-800 border border-yellow-300 p-3 rounded text-sm">
            ⚠️ Para obtener una guía de estudio de calidad, se recomienda subir contenido relevante en formato <strong>.pdf</strong>.
          </div>
        )}

        {/* Consejo adicional */}
        <div className="bg-blue-50 text-blue-800 text-sm p-4 rounded-md">
          💡 Consejo: Mientras más claro sea tu tema y palabras clave, mejores resultados obtendrás.
        </div>

        {/* Botón */}
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition flex items-center justify-center gap-2"
        >
          {loading ? "⏳ Generando..." : "🚀 Generar Guía"}
        </button>
      </form>

      {/* Mensaje general */}
      {message && (
        <div className="mt-6 bg-blue-50 p-5 rounded-lg shadow-sm text-blue-800">
          {message}
        </div>
      )}
    </div>
  );
}
