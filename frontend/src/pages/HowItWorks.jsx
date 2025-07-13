export default function HowItWorks() {
  return (
    <div className="max-w-2xl mx-auto p-8 bg-gray-50 rounded-xl shadow border border-gray-200 mt-10">
      {/* Título */}
      <h1 className="text-3xl font-bold text-blue-700 mb-2 text-center flex items-center justify-center gap-2">
        ¿Cómo funciona <span className="text-pink-600">StudyPlanner IA</span>?
      </h1>

      {/* Subtítulo */}
      <p className="text-center text-gray-500 mb-6 text-sm">
        Descubre cómo generar tu guía de estudio personalizada con IA en solo 4 pasos.
      </p>

      {/* Pasos */}
      <div className="space-y-6 text-sm text-gray-700">
        <div className="flex items-start gap-3">
          <span className="text-blue-700 text-xl">📄</span>
          <p>
            <strong>1. Sube tus documentos PDF:</strong> Añade archivos con el contenido que deseas estudiar para que la IA los analice.
          </p>
        </div>

        <div className="flex items-start gap-3">
          <span className="text-pink-600 text-xl">🧠</span>
          <p>
            <strong>2. Define tu objetivo de estudio:</strong> Escribe el tema, selecciona el nivel (básico, intermedio o avanzado) y agrega hasta 3 palabras clave.
          </p>
        </div>

        <div className="flex items-start gap-3">
          <span className="text-indigo-600 text-xl">⚙️</span>
          <p>
            <strong>3. La IA genera tu guía:</strong> El sistema construye un plan con módulos, objetivos, actividades prácticas y recursos útiles.
          </p>
        </div>

        <div className="flex items-start gap-3">
          <span className="text-purple-600 text-xl">🔗</span>
          <p>
            <strong>4. Accede a tu guía:</strong> Recibirás un enlace único para revisar, guardar o compartir tu guía cuando lo necesites.
          </p>
        </div>
      </div>

      {/* Botón CTA */}
      <div className="mt-10 text-center">
        <p className="text-sm text-gray-600 mb-2">¿Lista/o para comenzar?</p>
        <a
          href="/generar"
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition inline-block"
        >
          🚀 Generar mi Guía de Estudio
        </a>
      </div>
    </div>
  );
}
