import asyncio
import json
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
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
    version="2.6.0",  # Versión mejorada
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


class StudyGuideResponse(BaseModel):
    guide_id: str
    topic: str
    summary: str
    study_plan: str
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
                            "upload_time": datetime.now().isoformat(),
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
                                "search_time": datetime.now().isoformat(),
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
                "processed_at": datetime.now().isoformat(),
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
        model="gemini-2.5-flash",
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

    # Prompt mejorado y más específico
    prompt_template = """
    Eres un tutor experto en IA especializado en crear guías de estudio comprehensivas. 
    
    **IMPORTANTE**: Tu respuesta debe ser ÚNICAMENTE un objeto JSON válido, sin texto adicional, 
    sin marcadores de código, sin explicaciones antes o después del JSON.

    **Contexto de Información:**
    {context}

    **Tarea:**
    Crear una guía de estudio sobre "{input}" para nivel "{depth}" basándose en el contexto proporcionado.

    **Formato de Respuesta Requerido:**
    {{
      "summary": "Resumen detallado y comprehensivo del tema en mínimo 4 párrafos extensos, explicando conceptos clave, importancia, aplicaciones y fundamentos teóricos. Debe ser educativo y accesible para el nivel {depth}.",
      "study_plan": "Plan de estudio estructurado en formato Markdown con 4-6 módulos. Cada módulo debe incluir:\n### Módulo X: [Título]\n**Objetivos:**\n- Objetivo 1\n- Objetivo 2\n- Objetivo 3\n\n**Temas Clave:**\n- Tema 1\n- Tema 2\n- Tema 3\n\n**Actividades Prácticas:**\n- Actividad 1\n- Actividad 2\n- Actividad 3\n\n**Recursos Recomendados:**\n- Recurso 1\n- Recurso 2\n\n**Tiempo Estimado:** X horas\n\n"
    }}

    **Instrucciones Específicas:**
    1. El resumen debe ser substantivo y educativo, no superficial
    2. El plan de estudio debe ser práctico y accionable
    3. Adapta el contenido al nivel de profundidad especificado
    4. Incluye estimaciones de tiempo realistas
    5. Sugiere recursos y actividades específicas
    
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

        # Manejar diferentes tipos de respuesta
        if isinstance(answer_text, dict):
            guide_json = answer_text
        else:
            # Limpiar la respuesta de posibles marcadores
            cleaned_text = answer_text.strip()

            # Remover marcadores de código si existen
            if "```json" in cleaned_text:
                start = cleaned_text.find("```json") + 7
                end = cleaned_text.rfind("```")
                if end > start:
                    cleaned_text = cleaned_text[start:end].strip()
            elif "```" in cleaned_text:
                start = cleaned_text.find("```") + 3
                end = cleaned_text.rfind("```")
                if end > start:
                    cleaned_text = cleaned_text[start:end].strip()

            # Buscar el objeto JSON en el texto
            json_start = cleaned_text.find("{")
            json_end = cleaned_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                guide_json = json.loads(json_str)
            else:
                raise ValueError("No se encontró un objeto JSON válido en la respuesta")

        # Validar que los campos requeridos estén presentes
        required_fields = ["summary", "study_plan"]
        for field in required_fields:
            if field not in guide_json or not guide_json[field]:
                raise ValueError(f"Campo requerido '{field}' no encontrado o vacío")

        return {
            "summary": guide_json["summary"],
            "study_plan": guide_json["study_plan"],
            "sources_used": sorted(list(sources_used)),
            "metadata": {
                "topic": topic,
                "depth": depth,
                "sources_count": len(sources_used),
                "generated_at": datetime.now().isoformat(),
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
            "study_plan": f"## Plan de Estudio de Emergencia para: {topic}\n\n"
            f"**Nota:** Hubo un error en la generación automática. "
            f"Se recomienda intentar nuevamente o consultar las fuentes directamente.\n\n"
            f"### Módulo 1: Investigación Inicial\n"
            f"**Objetivos:**\n"
            f"- Investigar conceptos básicos de {topic}\n"
            f"- Identificar fuentes confiables\n"
            f"- Crear un plan de estudio personalizado\n\n"
            f"**Tiempo Estimado:** 2-3 horas\n\n"
            f"**Respuesta original del modelo:**\n"
            f"```\n{response.get('answer', 'No disponible')[:1000]}...\n```",
            "sources_used": sorted(list(sources_used)),
            "metadata": {
                "topic": topic,
                "depth": depth,
                "sources_count": len(sources_used),
                "generated_at": datetime.now().isoformat(),
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
        "version": "2.6.0",
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

        # Verificar conectividad de servicios
        services_status = {
            "vectorstore": {
                "status": "inicializado" if vectorstore else "no inicializado",
                "documents_count": doc_count,
                "persist_directory": PERSIST_DIRECTORY,
            },
            "embeddings": {
                "status": "inicializado" if embeddings_model else "no inicializado",
                "model": "models/embedding-001" if embeddings_model else None,
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
            "timestamp": datetime.now().isoformat(),
            "version": "2.6.0",
            "services": services_status,
            "configuration": {
                "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
                "supported_file_types": SUPPORTED_FILE_TYPES,
                "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


@app.post("/upload-pdfs/", response_model=UploadResponse)
async def upload_pdfs_endpoint(files: List[UploadFile] = File(...)):
    """Endpoint mejorado para subir PDFs con mejor validación y manejo de errores."""
    start_time = datetime.now()

    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos.")

    # Validar archivos antes de procesarlos
    valid_files = []
    errors = []

    for file in files:
        if not validate_file_type(file.filename):
            errors.append(f"Tipo no soportado: {file.filename}")
            continue

        if not validate_file_size(file):
            errors.append(f"Archivo muy grande: {file.filename}")
            continue

        valid_files.append(file)

    if not valid_files:
        raise HTTPException(
            status_code=400,
            detail=f"No hay archivos válidos para procesar. Errores: {errors}",
        )

    try:
        # Procesar archivos en el pool de threads
        documents = await asyncio.to_thread(process_and_load_pdfs, valid_files)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No se pudo extraer contenido de los PDFs válidos.",
            )

        # Añadir documentos a la base de vectores
        docs_added = await asyncio.to_thread(add_documents_to_vectorstore, documents)

        # Registrar en Firebase
        upload_id = f"upload_{uuid.uuid4().hex[:8]}"
        processing_time = (datetime.now() - start_time).total_seconds()

        upload_log = {
            "timestamp": datetime.now(),
            "files_processed": [f.filename for f in valid_files],
            "documents_extracted": len(documents),
            "chunks_added_to_db": docs_added,
            "processing_time_seconds": processing_time,
            "errors": errors if errors else None,
        }

        db.collection("uploads_log").document(upload_id).set(upload_log)

        response = UploadResponse(
            status="éxito",
            message=f"{len(valid_files)} PDF(s) procesados correctamente.",
            upload_id=upload_id,
            chunks_added=docs_added,
            processing_time=processing_time,
        )

        if errors:
            response.message += f" Advertencias: {len(errors)} archivos omitidos."

        return response

    except Exception as e:
        print(f"❌ [API] Error en upload_pdfs_endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error interno al procesar PDFs: {str(e)}"
        )


async def generate_guide_background_task(guide_id: str, request: StudyRequest):
    """Tarea en segundo plano mejorada para generar la guía."""
    print(f"⏳ [BackgroundTask] Iniciando tarea para la guía: {guide_id}")

    start_time = datetime.now()

    try:
        # Verificar servicios
        if not vectorstore or not tavily_client:
            raise RuntimeError(
                "Los servicios (VectorStore/Tavily) no se inicializaron correctamente."
            )

        # Actualizar estado a procesando con más detalles
        db.collection("study_guides").document(guide_id).update(
            {
                "status": "procesando",
                "processing_started_at": start_time,
                "current_step": "búsqueda_web",
            }
        )

        # Manejo inteligente de búsquedas web
        if request.search_queries:
            print("   -> 🌐 [BackgroundTask] Realizando búsquedas personalizadas...")
            search_docs = await asyncio.to_thread(
                search_online_sources, request.search_queries
            )
            if search_docs:
                await asyncio.to_thread(add_documents_to_vectorstore, search_docs)
        elif vectorstore._collection.count() == 0:
            print(
                "   -> ⚠️  [BackgroundTask] DB vacía. Generando búsquedas automáticas..."
            )
            # Búsquedas más inteligentes basadas en el tema y profundidad
            depth_mapping = {
                "básico": ["introducción", "conceptos básicos", "fundamentos"],
                "intermedio": ["guía completa", "conceptos clave", "aplicaciones"],
                "avanzado": [
                    "análisis avanzado",
                    "técnicas especializadas",
                    "investigación",
                ],
            }

            depth_terms = depth_mapping.get(request.depth, depth_mapping["intermedio"])
            default_queries = [f"{term} de {request.topic}" for term in depth_terms]

            search_docs = await asyncio.to_thread(
                search_online_sources, default_queries
            )
            if search_docs:
                await asyncio.to_thread(add_documents_to_vectorstore, search_docs)

        # Actualizar progreso
        db.collection("study_guides").document(guide_id).update(
            {"current_step": "generación_contenido"}
        )

        # Generar la guía
        print("   -> 🧠 [BackgroundTask] Generando contenido de la guía...")
        llm = get_llm()

        # Configuración optimizada del retriever
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",  # Máxima diversidad marginal
            search_kwargs={
                "k": 25,  # Más documentos para mejor contexto
                "fetch_k": 50,  # Más candidatos para MMR
                "lambda_mult": 0.7,  # Balance entre relevancia y diversidad
            },
        )

        guide_data = await asyncio.to_thread(
            generate_study_guide, request.topic, base_retriever, llm, request.depth
        )

        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds()

        # Actualizar documento con el resultado completo
        db.collection("study_guides").document(guide_id).update(
            {
                "status": "completado",
                "summary": guide_data["summary"],
                "study_plan": guide_data["study_plan"],
                "sources_used": guide_data["sources_used"],
                "completed_at": datetime.now(),
                "processing_time_seconds": processing_time,
                "metadata": guide_data.get("metadata", {}),
                "current_step": "completado",
            }
        )

        print(
            f"✅ [BackgroundTask] Tarea completada para la guía: {guide_id} ({processing_time:.2f}s)"
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)

        print(f"❌ [BackgroundTask] Error en tarea para guía {guide_id}: {error_msg}")

        # Actualizar con información de error detallada
        db.collection("study_guides").document(guide_id).update(
            {
                "status": "error",
                "error_message": error_msg,
                "error_type": type(e).__name__,
                "completed_at": datetime.now(),
                "processing_time_seconds": processing_time,
                "current_step": "error",
            }
        )


@app.post("/generate-study-guide-async/")
async def generate_study_guide_async_endpoint(
    request: StudyRequest, background_tasks: BackgroundTasks
):
    """Endpoint mejorado para generar guías de estudio de forma asíncrona."""
    print(f"⚡️ [API-Async] Solicitud para generar guía sobre: '{request.topic}'")

    # Generar ID más legible
    guide_id = (
        f"guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )

    # Crear documento inicial con más información
    initial_doc = {
        "status": "iniciado",
        "topic": request.topic,
        "depth": request.depth,
        "search_queries": request.search_queries,
        "created_at": datetime.now(),
        "guide_id": guide_id,
        "version": "2.6.0",
        "estimated_completion_minutes": 2,  # Estimación basada en experiencia
        "current_step": "inicializando",
    }

    db.collection("study_guides").document(guide_id).set(initial_doc)

    # Añadir tarea en segundo plano
    background_tasks.add_task(generate_guide_background_task, guide_id, request)

    return {
        "message": "Solicitud recibida. La guía se está generando en segundo plano.",
        "guide_id": guide_id,
        "status_url": f"/study-guide/{guide_id}",
        "estimated_completion": "1-3 minutos",
        "topic": request.topic,
        "depth": request.depth,
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
        doc_ref = db.collection("study_guides").document(guide_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(
                status_code=404, detail=f"Guía con ID '{guide_id}' no encontrada."
            )

        data = doc.to_dict()
        status = data.get("status")
        current_step = data.get("current_step", "desconocido")

        # Estado: procesando o iniciado
        if status in ["procesando", "iniciado"]:
            processing_time = None
            if "processing_started_at" in data:
                start_time = data["processing_started_at"]
                if isinstance(start_time, datetime):
                    processing_time = (datetime.now() - start_time).total_seconds()

            return JSONResponse(
                status_code=202,
                content={
                    "status": status,
                    "current_step": current_step,
                    "message": f"La guía se está procesando. Paso actual: {current_step}",
                    "guide_id": guide_id,
                    "topic": data.get("topic"),
                    "processing_time_seconds": processing_time,
                    "estimated_remaining_minutes": max(
                        0, 2 - (processing_time / 60 if processing_time else 0)
                    ),
                },
            )

        # Estado: error
        elif status == "error":
            error_details = {
                "guide_id": guide_id,
                "status": "error",
                "error_message": data.get("error_message", "Error desconocido"),
                "error_type": data.get("error_type", "Unknown"),
                "processing_time_seconds": data.get("processing_time_seconds"),
                "topic": data.get("topic"),
                "created_at": data.get("created_at").isoformat()
                if data.get("created_at")
                else None,
            }

            raise HTTPException(
                status_code=500,
                detail=f"Error al generar la guía: {error_details['error_message']}",
            )

        # Estado: completado
        elif status == "completado":
            return StudyGuideResponse(
                guide_id=guide_id,
                topic=data.get("topic", ""),
                summary=data.get("summary", ""),
                study_plan=data.get("study_plan", ""),
                sources_used=data.get("sources_used", []),
                created_at=data.get("created_at").isoformat()
                if data.get("created_at")
                else None,
                completed_at=data.get("completed_at").isoformat()
                if data.get("completed_at")
                else None,
            )

        # Estado desconocido
        else:
            raise HTTPException(status_code=500, detail=f"Estado desconocido: {status}")

    except HTTPException:
        raise  # Re-lanzar HTTPExceptions
    except Exception as e:
        print(f"❌ [API] Error en get_study_guide_status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error interno al consultar la guía: {str(e)}"
        )


@app.get("/guides/")
async def list_recent_guides(limit: int = 10):
    """Nuevo endpoint para listar guías recientes."""
    try:
        guides_ref = (
            db.collection("study_guides")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )

        guides = []
        for doc in guides_ref.stream():
            data = doc.to_dict()
            guides.append(
                {
                    "guide_id": doc.id,
                    "topic": data.get("topic"),
                    "status": data.get("status"),
                    "depth": data.get("depth"),
                    "created_at": data.get("created_at").isoformat()
                    if data.get("created_at")
                    else None,
                    "completed_at": data.get("completed_at").isoformat()
                    if data.get("completed_at")
                    else None,
                    "processing_time_seconds": data.get("processing_time_seconds"),
                }
            )

        return {"guides": guides, "total_returned": len(guides)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al consultar guías: {str(e)}"
        )


@app.delete("/admin/reset-knowledge-base/")
async def reset_knowledge_base():
    """Endpoint mejorado para reiniciar la base de conocimientos."""
    print("🚨 [Admin] Reiniciando la base de conocimientos...")

    try:
        # Backup de estadísticas antes de borrar
        doc_count = vectorstore._collection.count() if vectorstore else 0

        # Borrar directorio de persistencia
        if os.path.exists(PERSIST_DIRECTORY):
            await asyncio.to_thread(shutil.rmtree, PERSIST_DIRECTORY)
            print(
                f"   -> 🗑️  Eliminados {doc_count} documentos del directorio persistente"
            )

        # Reinicializar servicios
        await asyncio.to_thread(initialize_services)

        # Registrar la acción en Firebase
        reset_log = {
            "timestamp": datetime.now(),
            "action": "knowledge_base_reset",
            "documents_removed": doc_count,
            "admin_action": True,
        }

        db.collection("admin_actions").add(reset_log)

        return {
            "status": "éxito",
            "message": f"Base de conocimientos reiniciada. {doc_count} documentos eliminados.",
            "timestamp": datetime.now().isoformat(),
            "new_document_count": vectorstore._collection.count() if vectorstore else 0,
        }

    except Exception as e:
        print(f"❌ [Admin] Error al reiniciar: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al reiniciar la base de conocimientos: {str(e)}",
        )


@app.get("/admin/stats/")
async def get_admin_statistics():
    """Nuevo endpoint para obtener estadísticas administrativas."""
    try:
        # Estadísticas de la base de vectores
        vector_stats = {
            "total_documents": vectorstore._collection.count() if vectorstore else 0,
            "persist_directory": PERSIST_DIRECTORY,
            "collection_name": CHROMA_COLLECTION_NAME,
        }

        # Estadísticas de Firebase (últimas 24 horas)
        yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Contar guías por estado
        guides_ref = db.collection("study_guides").where("created_at", ">=", yesterday)
        guides_by_status = {"completado": 0, "error": 0, "procesando": 0, "iniciado": 0}
        total_processing_time = 0
        completed_count = 0

        for doc in guides_ref.stream():
            data = doc.to_dict()
            status = data.get("status", "unknown")
            if status in guides_by_status:
                guides_by_status[status] += 1

            if status == "completado" and "processing_time_seconds" in data:
                total_processing_time += data["processing_time_seconds"]
                completed_count += 1

        # Contar uploads
        uploads_ref = db.collection("uploads_log").where("timestamp", ">=", yesterday)
        upload_count = sum(1 for _ in uploads_ref.stream())

        avg_processing_time = (
            total_processing_time / completed_count if completed_count > 0 else 0
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "vector_database": vector_stats,
            "last_24_hours": {
                "guides_by_status": guides_by_status,
                "total_guides": sum(guides_by_status.values()),
                "upload_sessions": upload_count,
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "success_rate": (
                    round(
                        guides_by_status["completado"]
                        / sum(guides_by_status.values())
                        * 100,
                        2,
                    )
                    if sum(guides_by_status.values()) > 0
                    else 0
                ),
            },
            "system_info": {
                "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
                "supported_file_types": SUPPORTED_FILE_TYPES,
                "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al obtener estadísticas: {str(e)}"
        )


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
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path) if hasattr(request, "url") else "unknown",
        },
    )


if __name__ == "__main__":
    import uvicorn

    print("🚀 Iniciando servidor mejorado con Uvicorn...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Deshabilitado para producción
        access_log=True,
        log_level="info",
    )

