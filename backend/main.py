import asyncio
import json
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import firebase_admin
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from firebase_admin import credentials, firestore

# --- LangChain Imports ---
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, validator

# --- Tavily Search API Client ---
from tavily import TavilyClient

# --- Carga de variables de entorno desde .env ---
load_dotenv()

# --- Inicialización de Firebase Admin SDK ---
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH", "firebase.json")
if not firebase_admin._apps:
    try:
        print("🔍 [Firebase] Inicializando Firebase Admin SDK...")
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        print("✅ [Firebase] Firebase Admin SDK inicializado correctamente.")
    except Exception as e:
        print(f"❌ [Firebase] Error al inicializar Firebase: {e}")
        raise

db = firestore.client()

# --- Variables Globales y Configuración ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

PERSIST_DIRECTORY = "./chroma_db_persistente"
CHROMA_COLLECTION_NAME = "guias_de_estudio_collection"

# Configuración mejorada
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FILE_TYPES = [".pdf"]
MAX_CONCURRENT_TASKS = 5
EMBEDDING_RETRY_ATTEMPTS = 3
WEB_SEARCH_RETRY_ATTEMPTS = 2

embeddings_model: Optional[GoogleGenerativeAIEmbeddings] = None
vectorstore: Optional[Chroma] = None
tavily_client: Optional[TavilyClient] = None
thread_pool: Optional[ThreadPoolExecutor] = None


# --- Ciclo de vida de la aplicación (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona la inicialización de servicios al arrancar la aplicación."""
    global thread_pool
    print("🚀 [FastAPI] Iniciando la aplicación...")

    # Inicializar pool de threads para operaciones CPU-intensivas
    thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)

    try:
        await asyncio.to_thread(initialize_services)
        yield
    finally:
        print("👋 [FastAPI] Apagando la aplicación...")
        if thread_pool:
            thread_pool.shutdown(wait=True)


def initialize_services():
    """Inicializa los servicios principales (modelos, vector store, cliente de búsqueda)."""
    global embeddings_model, vectorstore, tavily_client
    print("🔍 [Servicios] Inicializando servicios clave...")

    # Validación de variables de entorno
    if not GOOGLE_API_KEY:
        raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")
    if not TAVILY_API_KEY:
        raise ValueError("La variable de entorno TAVILY_API_KEY no está configurada.")

    # Inicialización con reintentos
    for attempt in range(EMBEDDING_RETRY_ATTEMPTS):
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY,
                request_options={"timeout": 60},  # Timeout más largo
            )
            print("✅ [Embeddings] Modelo de embeddings cargado.")
            break
        except Exception as e:
            print(f"❌ [Embeddings] Intento {attempt + 1} fallido: {e}")
            if attempt == EMBEDDING_RETRY_ATTEMPTS - 1:
                raise
            asyncio.sleep(2**attempt)  # Backoff exponencial

    try:
        # Crear directorio si no existe
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings_model,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        doc_count = vectorstore._collection.count()
        print(f"✅ [VectorStore] ChromaDB inicializada en '{PERSIST_DIRECTORY}'.")
        print(f"ℹ️  [VectorStore] Documentos en DB: {doc_count}")
    except Exception as e:
        print(f"❌ [VectorStore] Error al inicializar ChromaDB: {e}")
        raise

    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        print("✅ [Tavily] Cliente de búsqueda web Tavily inicializado.")
    except Exception as e:
        print(f"❌ [Tavily] Error al inicializar el cliente de Tavily: {e}")
        raise


# --- Inicialización de la App FastAPI ---
app = FastAPI(
    title="🤖 Agente IA para Guías de Estudio",
    description="Un agente inteligente para crear resúmenes y planes de estudio a partir de PDFs y fuentes online.",
    version="2.8.0",  # Versión mejorada
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Modelos Pydantic Mejorados ---
class StudyRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200, description="Tema de estudio")
    search_queries: Optional[List[str]] = Field(
        default=[], description="Consultas de búsqueda específicas", max_items=10
    )
    depth: str = Field(
        default="intermedio", description="Nivel de profundidad del estudio"
    )

    @validator("depth")
    def validate_depth(cls, v):
        allowed_depths = ["básico", "intermedio", "avanzado"]
        if v not in allowed_depths:
            raise ValueError(f"Profundidad debe ser una de: {allowed_depths}")
        return v

    @validator("search_queries")
    def validate_search_queries(cls, v):
        if v:
            for query in v:
                if len(query.strip()) < 3:
                    raise ValueError("Cada consulta debe tener al menos 3 caracteres")
        return v


class StudyPlanModule(BaseModel):
    module_number: int = Field(..., description="Número secuencial del módulo")
    title: str = Field(..., description="Título del módulo")
    objectives: List[str] = Field(..., description="Lista de objetivos de aprendizaje")
    key_topics: List[str] = Field(..., description="Lista de temas clave a cubrir")
    practical_activities: List[str] = Field(
        ..., description="Lista de actividades prácticas sugeridas"
    )
    recommended_resources: List[str] = Field(
        ..., description="Lista de recursos recomendados (links, libros, etc.)"
    )
    estimated_time: str = Field(
        ..., description="Tiempo estimado para completar el módulo (e.g., '2 horas')"
    )


class StudyGuideResponse(BaseModel):
    guide_id: str
    topic: str
    summary: str
    study_plan: List[StudyPlanModule]  # Estructura de plan de estudio mejorada
    sources_used: List[str]
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    message: str
    upload_id: str
    chunks_added: int
    processing_time: Optional[float] = None


# --- Funciones de Validación ---
def validate_file_size(file: UploadFile) -> bool:
    """Valida el tamaño del archivo."""
    if hasattr(file, "size") and file.size and file.size > MAX_FILE_SIZE:
        return False
    return True


def validate_file_type(filename: str) -> bool:
    """Valida el tipo de archivo."""
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_FILE_TYPES)


# --- Funciones de Procesamiento de Documentos Mejoradas ---
def process_and_load_pdfs(files: List[UploadFile]) -> List[Document]:
    """Procesa archivos PDF subidos, extrayendo su contenido con mejor manejo de errores."""
    print(f"📄 [PDF] Procesando {len(files)} archivo(s) PDF...")
    documents = []
    processing_errors = []

    for file in files:
        try:
            # Validaciones
            if not validate_file_type(file.filename):
                print(f"⚠️  [PDF] Tipo de archivo no soportado: {file.filename}")
                processing_errors.append(f"Tipo no soportado: {file.filename}")
                continue

            if not validate_file_size(file):
                print(f"⚠️  [PDF] Archivo muy grande: {file.filename}")
                processing_errors.append(f"Archivo muy grande: {file.filename}")
                continue

            # Procesamiento con contexto de manejo de archivos temporales
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = file.file.read()
                if len(content) == 0:
                    print(f"⚠️  [PDF] Archivo vacío: {file.filename}")
                    processing_errors.append(f"Archivo vacío: {file.filename}")
                    continue

                tmp_file.write(content)
                tmp_path = tmp_file.name

            try:
                loader = PyPDFLoader(tmp_path)
                pdf_docs = loader.load()

                if not pdf_docs:
                    print(f"⚠️  [PDF] No se pudo extraer contenido: {file.filename}")
                    processing_errors.append(
                        f"Sin contenido extraíble: {file.filename}"
                    )
                    continue

                # Enriquecer metadatos
                for doc in pdf_docs:
                    doc.metadata.update(
                        {
                            "source": f"PDF: {file.filename}",
                            "type": "pdf_upload",
                            "upload_time": datetime.now(timezone.utc).isoformat(),
                            "file_size": len(content),
                            "page_count": len(pdf_docs),
                        }
                    )

                documents.extend(pdf_docs)
                print(
                    f"✅ [PDF] Procesado exitosamente: {file.filename} ({len(pdf_docs)} páginas)"
                )

            except Exception as e:
                print(f"❌ [PDF] Error al procesar {file.filename}: {e}")
                processing_errors.append(
                    f"Error de procesamiento en {file.filename}: {str(e)}"
                )
            finally:
                # Limpiar archivo temporal
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            print(f"❌ [PDF] Error general con {file.filename}: {e}")
            processing_errors.append(f"Error general en {file.filename}: {str(e)}")
        finally:
            if hasattr(file, "file"):
                file.file.close()

    print(f"✅ [PDF] {len(documents)} documentos extraídos de los PDFs.")
    if processing_errors:
        print(f"⚠️  [PDF] Errores de procesamiento: {processing_errors}")

    return documents


def search_online_sources(queries: List[str], max_results: int = 7) -> List[Document]:
    """Realiza búsquedas en la web usando Tavily con reintentos."""
    if not tavily_client:
        raise ValueError("El cliente de Tavily no está inicializado.")

    print(f"🌐 [WebSearch] Buscando en la web para {len(queries)} consulta(s)...")
    all_documents = []

    for query in queries:
        for attempt in range(WEB_SEARCH_RETRY_ATTEMPTS):
            try:
                response = tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_results,
                    include_answer=True,
                    include_raw_content=True,
                )

                if "results" in response:
                    for result in response["results"]:
                        # Filtrar resultados vacíos o muy cortos
                        content = result.get("content", "").strip()
                        if len(content) < 50:  # Contenido muy corto
                            continue

                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": result.get("url", "fuente desconocida"),
                                "title": result.get("title", "Título Desconocido"),
                                "type": "web_search",
                                "query": query,
                                "search_time": datetime.now(timezone.utc).isoformat(),
                                "score": result.get("score", 0.0),
                            },
                        )
                        all_documents.append(doc)

                print(f"✅ [WebSearch] Búsqueda exitosa para: '{query}'")
                break  # Salir del bucle de reintentos si fue exitoso

            except Exception as e:
                print(
                    f"❌ [WebSearch] Intento {attempt + 1} fallido para '{query}': {e}"
                )
                if attempt == WEB_SEARCH_RETRY_ATTEMPTS - 1:
                    print(
                        f"❌ [WebSearch] Falló definitivamente la búsqueda para: '{query}'"
                    )
                else:
                    asyncio.sleep(1)  # Esperar antes del siguiente intento

    # Eliminar documentos duplicados basándose en URL
    seen_urls = set()
    unique_documents = []
    for doc in all_documents:
        url = doc.metadata.get("source")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_documents.append(doc)

    print(
        f"✅ [WebSearch] {len(unique_documents)} documentos únicos encontrados en la web."
    )
    return unique_documents


def split_documents_into_chunks(documents: List[Document]) -> List[Document]:
    """Divide documentos en fragmentos más pequeños con configuración optimizada."""
    if not documents:
        return []

    # Configuración adaptativa basada en el tipo de documento
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Reducido para mejor contexto
        chunk_overlap=200,  # Reducido para evitar redundancia excesiva
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    # Enriquecer metadatos de chunks
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "chunk_id": f"chunk_{i}",
                "chunk_length": len(chunk.page_content),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return chunks


def add_documents_to_vectorstore(documents: List[Document]) -> int:
    """Añade documentos a la base de datos de vectores con mejor manejo de errores."""
    if not vectorstore:
        raise ValueError("La base de datos de vectores no está inicializada.")

    if not documents:
        return 0

    chunks = split_documents_into_chunks(documents)
    if not chunks:
        return 0

    print(f"➕ [VectorStore] Añadiendo {len(chunks)} nuevos fragmentos a ChromaDB...")

    try:
        # Procesar en lotes para evitar problemas de memoria
        batch_size = 100
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectorstore.add_documents(batch)
            total_added += len(batch)
            print(f"   -> Lote {i // batch_size + 1}: {len(batch)} fragmentos añadidos")

        print(f"✅ [VectorStore] Total de fragmentos añadidos: {total_added}")
        return total_added

    except Exception as e:
        print(f"❌ [VectorStore] Error al añadir documentos: {e}")
        raise


# --- Lógica Principal del Agente (Generación de Guía) Mejorada ---
def get_llm():
    """Inicializa y devuelve el modelo de lenguaje de Google (Gemini) con configuración optimizada."""
    return GoogleGenerativeAI(
        # model="gemini-2.5-flash-lite-preview-06-17",
        model="gemini-2.5-flash-lite-preview-06-17",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,  # Ligeramente más creativo
        max_output_tokens=8192,
        convert_system_message_to_human=True,
        model_kwargs={
            "response_mime_type": "application/json",
            "candidate_count": 1,
            "stop_sequences": [],
        },
        request_timeout=120,  # Timeout más largo
    )


def generate_study_guide(topic: str, base_retriever, llm, depth: str) -> Dict[str, Any]:
    """Orquesta el proceso RAG para crear la guía, con mejor manejo de respuestas JSON."""
    print(
        f"🧠 [RAG] Iniciando generación de guía para '{topic}' (profundidad: {depth})."
    )

    # Configurar retriever con parámetros optimizados
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    # Prompt mejorado para una salida JSON estructurada
    prompt_template = """
    Eres un tutor experto en IA especializado en crear guías de estudio comprehensivas. 
    
    **IMPORTANTE**: Tu respuesta debe ser ÚNICAMENTE un objeto JSON válido, sin texto adicional, 
    sin marcadores de código, sin explicaciones antes o después del JSON.

    **Contexto de Información:**
    {context}

    **Tarea:**
    Crear una guía de estudio sobre "{input}" para nivel "{depth}" basándose en el contexto proporcionado.

    **Formato de Respuesta Requerido (JSON):**
    {{
      "summary": "Resumen detallado y comprehensivo del tema en mínimo 4 párrafos extensos, explicando conceptos clave, importancia, aplicaciones y fundamentos teóricos. Debe ser educativo y accesible para el nivel {depth}.",
      "study_plan": [
        {{
          "module_number": 1,
          "title": "Título del Módulo 1",
          "objectives": ["Objetivo 1.1", "Objetivo 1.2", "Objetivo 1.3"],
          "key_topics": ["Tema clave 1.1", "Tema clave 1.2", "Tema clave 1.3"],
          "practical_activities": ["Actividad práctica 1.1", "Actividad práctica 1.2"],
          "recommended_resources": ["Recurso recomendado 1.1", "Recurso recomendado 1.2"],
          "estimated_time": "X horas"
        }},
        {{
          "module_number": 2,
          "title": "Título del Módulo 2",
          "objectives": ["Objetivo 2.1", "Objetivo 2.2"],
          "key_topics": ["Tema clave 2.1", "Tema clave 2.2"],
          "practical_activities": ["Actividad práctica 2.1"],
          "recommended_resources": ["Recurso recomendado 2.1"],
          "estimated_time": "Y horas"
        }}
      ]
    }}

    **Instrucciones Específicas:**
    1. El resumen debe ser substantivo y educativo.
    2. El 'study_plan' debe ser un array de objetos JSON, con 4-6 módulos. Cada objeto debe tener las claves: 'module_number', 'title', 'objectives', 'key_topics', 'practical_activities', 'recommended_resources', y 'estimated_time'.
    3. Adapta el contenido al nivel de profundidad especificado.
    4. Incluye estimaciones de tiempo realistas.
    5. Sugiere recursos y actividades específicas.
    
    Responde SOLO con el objeto JSON:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "input", "depth"]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("   -> 💬 [RAG] Invocando la cadena de generación...")
    try:
        response = retrieval_chain.invoke({"input": topic, "depth": depth})
        print("   -> ✅ [RAG] Respuesta recibida del LLM.")
    except Exception as e:
        print(f"   -> ❌ [RAG] Error al invocar la cadena: {e}")
        raise

    # Extraer fuentes utilizadas
    sources_used = set()
    for doc in response.get("context", []):
        source = doc.metadata.get("source", "Fuente desconocida")
        if source != "Fuente desconocida":
            sources_used.add(source)

    # Procesamiento mejorado de la respuesta JSON
    try:
        answer_text = response.get("answer", "{}")

        if isinstance(answer_text, dict):
            guide_json = answer_text
        else:
            cleaned_text = (
                answer_text.strip().removeprefix("```json").removesuffix("```").strip()
            )
            guide_json = json.loads(cleaned_text)

        required_fields = ["summary", "study_plan"]
        for field in required_fields:
            if field not in guide_json or not guide_json[field]:
                raise ValueError(
                    f"Campo requerido '{field}' no encontrado o vacío en la respuesta del LLM."
                )

        if not isinstance(guide_json["study_plan"], list):
            raise ValueError("El campo 'study_plan' debe ser una lista (array).")

        # Validar cada módulo usando el modelo Pydantic para robustez adicional
        validated_plan = [
            StudyPlanModule(**module) for module in guide_json["study_plan"]
        ]

        return {
            "summary": guide_json["summary"],
            "study_plan": validated_plan,
            "sources_used": sorted(list(sources_used)),
            "metadata": {
                "topic": topic,
                "depth": depth,
                "sources_count": len(sources_used),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        print(f"❌ [RAG] Error al procesar respuesta JSON: {e}")
        print(f"   -> Respuesta cruda: {response.get('answer', 'N/A')[:500]}...")

        # Respuesta de fallback más informativa
        fallback_response = {
            "summary": f"Error al procesar la respuesta del modelo para el tema '{topic}'. "
            f"El modelo no generó una respuesta en el formato JSON esperado. "
            f"Error específico: {str(e)}",
            "study_plan": [],  # Devuelve una lista vacía para cumplir con el nuevo esquema
            "sources_used": sorted(list(sources_used)),
            "metadata": {
                "topic": topic,
                "depth": depth,
                "sources_count": len(sources_used),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            },
        }

        return fallback_response


# --- Endpoints de la API Mejorados ---
@app.get("/")
async def root():
    return {
        "message": "Bienvenido al Agente IA para Guías de Estudio",
        "status": "operativo",
        "version": "2.8.0",
        "endpoints": {
            "upload": "/upload-pdfs/",
            "generate_async": "/generate-study-guide-async/",
            "status": "/study-guide/{guide_id}",
            "system_status": "/status",
        },
    }


@app.get("/status")
async def get_status():
    """Endpoint mejorado de estado del sistema."""
    try:
        doc_count = vectorstore._collection.count() if vectorstore else 0
        services_status = {
            "vectorstore": {
                "status": "inicializado" if vectorstore else "no inicializado",
                "documents_count": doc_count,
            },
            "embeddings": {
                "status": "inicializado" if embeddings_model else "no inicializado"
            },
            "tavily_search": {
                "status": "inicializado" if tavily_client else "no inicializado"
            },
            "firebase": {
                "status": "inicializado" if firebase_admin._apps else "no inicializado"
            },
        }

        return {
            "status": "operativo",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.8.0",
            "services": services_status,
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }


@app.post("/upload-pdfs/", response_model=UploadResponse)
async def upload_pdfs_endpoint(files: List[UploadFile] = File(...)):
    """Endpoint mejorado para subir PDFs con mejor validación y manejo de errores."""
    start_time = datetime.now(timezone.utc)

    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos.")

    valid_files = [
        f for f in files if validate_file_type(f.filename) and validate_file_size(f)
    ]
    if not valid_files:
        raise HTTPException(
            status_code=400, detail="Ninguno de los archivos es válido."
        )

    try:
        documents = await asyncio.to_thread(process_and_load_pdfs, valid_files)
        if not documents:
            raise HTTPException(
                status_code=400, detail="No se pudo extraer contenido de los PDFs."
            )

        docs_added = await asyncio.to_thread(add_documents_to_vectorstore, documents)
        upload_id = f"upload_{uuid.uuid4().hex[:8]}"
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        db.collection("uploads_log").document(upload_id).set(
            {
                "timestamp": datetime.now(timezone.utc),
                "files_processed": [f.filename for f in valid_files],
                "chunks_added_to_db": docs_added,
                "processing_time_seconds": processing_time,
            }
        )

        return UploadResponse(
            status="éxito",
            message=f"{len(valid_files)} PDF(s) procesados correctamente.",
            upload_id=upload_id,
            chunks_added=docs_added,
            processing_time=processing_time,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error interno al procesar PDFs: {e}"
        )


async def generate_guide_background_task(guide_id: str, request: StudyRequest):
    """Tarea en segundo plano mejorada para generar la guía."""
    print(f"⏳ [BackgroundTask] Iniciando tarea para la guía: {guide_id}")
    start_time = datetime.now(timezone.utc)

    try:
        if not vectorstore or not tavily_client:
            raise RuntimeError(
                "Los servicios (VectorStore/Tavily) no se inicializaron."
            )

        db.collection("study_guides").document(guide_id).update(
            {
                "status": "procesando",
                "processing_started_at": start_time,
                "current_step": "búsqueda_web",
            }
        )

        search_queries = request.search_queries
        if not search_queries and vectorstore._collection.count() == 0:
            depth_terms = {
                "básico": "introducción",
                "intermedio": "guía completa",
                "avanzado": "análisis avanzado",
            }
            search_queries = [
                f"{depth_terms.get(request.depth, 'guía')} de {request.topic}"
            ]

        if search_queries:
            search_docs = await asyncio.to_thread(search_online_sources, search_queries)
            if search_docs:
                await asyncio.to_thread(add_documents_to_vectorstore, search_docs)

        db.collection("study_guides").document(guide_id).update(
            {"current_step": "generación_contenido"}
        )

        llm = get_llm()
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 25, "fetch_k": 50, "lambda_mult": 0.7},
        )
        guide_data = await asyncio.to_thread(
            generate_study_guide, request.topic, base_retriever, llm, request.depth
        )

        db.collection("study_guides").document(guide_id).update(
            {
                "status": "completado",
                "summary": guide_data["summary"],
                "study_plan": [
                    module.dict() for module in guide_data.get("study_plan", [])
                ],
                "sources_used": guide_data["sources_used"],
                "completed_at": datetime.now(timezone.utc),
                "processing_time_seconds": (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
                "metadata": guide_data.get("metadata", {}),
                "current_step": "completado",
            }
        )
        print(f"✅ [BackgroundTask] Tarea completada para la guía: {guide_id}")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ [BackgroundTask] Error en tarea para guía {guide_id}: {error_msg}")
        db.collection("study_guides").document(guide_id).update(
            {
                "status": "error",
                "error_message": error_msg,
                "completed_at": datetime.now(timezone.utc),
                "current_step": "error",
            }
        )


@app.post("/generate-study-guide-async/")
async def generate_study_guide_async_endpoint(
    request: StudyRequest, background_tasks: BackgroundTasks
):
    """Endpoint mejorado para generar guías de estudio de forma asíncrona."""
    guide_id = f"guide_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    initial_doc = {
        "status": "iniciado",
        "topic": request.topic,
        "depth": request.depth,
        "created_at": datetime.now(timezone.utc),
        "guide_id": guide_id,
        "current_step": "inicializando",
    }
    db.collection("study_guides").document(guide_id).set(initial_doc)

    background_tasks.add_task(generate_guide_background_task, guide_id, request)

    return {
        "message": "Solicitud recibida. La guía se está generando en segundo plano.",
        "guide_id": guide_id,
        "status_url": f"/study-guide/{guide_id}",
    }


@app.get(
    "/study-guide/{guide_id}",
    responses={
        200: {"description": "Guía completada", "model": StudyGuideResponse},
        202: {"description": "Procesando"},
        404: {"description": "Guía no encontrada"},
        500: {"description": "Error en procesamiento"},
    },
)
async def get_study_guide_status(guide_id: str):
    """Endpoint mejorado para consultar el estado y resultado de una guía."""
    try:
        doc = db.collection("study_guides").document(guide_id).get()
        if not doc.exists:
            raise HTTPException(
                status_code=404, detail=f"Guía con ID '{guide_id}' no encontrada."
            )

        data = doc.to_dict()
        status = data.get("status")

        if status in ["procesando", "iniciado"]:
            processing_time = None
            if "processing_started_at" in data and isinstance(
                data["processing_started_at"], datetime
            ):
                processing_time = (
                    datetime.now(timezone.utc) - data["processing_started_at"]
                ).total_seconds()

            return JSONResponse(
                status_code=202,
                content={
                    "status": status,
                    "current_step": data.get("current_step", "desconocido"),
                    "message": "La guía se está procesando.",
                    "processing_time_seconds": processing_time,
                },
            )

        elif status == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Error al generar la guía: {data.get('error_message', 'Error desconocido')}",
            )

        elif status == "completado":
            # Añadir robustez para manejar datos antiguos o malformados en Firestore
            raw_plan = data.get("study_plan", [])
            if not isinstance(raw_plan, list):
                print(
                    f"⚠️  [API] Corrigiendo 'study_plan' inválido para la guía {guide_id}. Se usará una lista vacía."
                )
                final_plan = []
            else:
                final_plan = raw_plan

            return StudyGuideResponse(
                guide_id=guide_id,
                topic=data.get("topic", ""),
                summary=data.get("summary", ""),
                study_plan=final_plan,
                sources_used=data.get("sources_used", []),
                created_at=data.get("created_at").isoformat()
                if data.get("created_at")
                else None,
                completed_at=data.get("completed_at").isoformat()
                if data.get("completed_at")
                else None,
            )

        else:
            raise HTTPException(status_code=500, detail=f"Estado desconocido: {status}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [API] Error en get_study_guide_status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error interno al consultar la guía: {e}"
        )


@app.get("/guides/")
async def list_recent_guides(limit: int = 10):
    """Nuevo endpoint para listar guías recientes."""
    guides_ref = (
        db.collection("study_guides")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    guides = [doc.to_dict() for doc in guides_ref.stream()]
    for guide in guides:
        if "created_at" in guide and isinstance(guide["created_at"], datetime):
            guide["created_at"] = guide["created_at"].isoformat()
        if "completed_at" in guide and isinstance(guide["completed_at"], datetime):
            guide["completed_at"] = guide["completed_at"].isoformat()
    return {"guides": guides}


@app.delete("/admin/reset-knowledge-base/")
async def reset_knowledge_base():
    """Endpoint mejorado para reiniciar la base de conocimientos."""
    print("🚨 [Admin] Reiniciando la base de conocimientos...")
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            await asyncio.to_thread(shutil.rmtree, PERSIST_DIRECTORY)
        await asyncio.to_thread(initialize_services)
        return {"status": "éxito", "message": "Base de conocimientos reiniciada."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reiniciar: {e}")


# --- Manejo de Errores Global ---
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejo global de excepciones mejorado."""
    print(f"❌ [Global] Excepción no manejada: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "type": type(exc).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    print("🚀 Iniciando servidor mejorado con Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
