"""Diccionarios y listas usados por el análisis determinista.

No pretende ser exhaustivo: cubre el camino sin LLM y sirve de apoyo al
detector de secciones y al extractor de keywords.
"""

from __future__ import annotations

from typing import Dict, List, Set

# --- Secciones del CV -----------------------------------------------------

#: Encabezados que un ATS reconoce sin ambigüedad, en ES y EN.
SECTION_ALIASES: Dict[str, List[str]] = {
    "experiencia": [
        "experiencia", "experiencia laboral", "experiencia profesional",
        "historial laboral", "trayectoria profesional", "empleo",
        "work experience", "professional experience", "employment history",
        "experience", "work history",
    ],
    "educacion": [
        "educacion", "educación", "formacion", "formación",
        "formacion academica", "formación académica", "estudios",
        "education", "academic background", "academic education",
    ],
    "skills": [
        "habilidades", "competencias", "aptitudes", "conocimientos",
        "conocimientos tecnicos", "conocimientos técnicos",
        "skills", "technical skills", "core competencies", "hard skills",
        "tecnologias", "tecnologías", "stack tecnologico", "stack tecnológico",
    ],
    "certificaciones": [
        "certificaciones", "certificados", "cursos",
        "certifications", "certificates", "courses", "licenses",
    ],
    "idiomas": ["idiomas", "languages", "lenguas"],
    "resumen": [
        "resumen", "perfil", "perfil profesional", "sobre mi", "sobre mí",
        "objetivo", "extracto",
        "summary", "professional summary", "profile", "objective", "about me",
    ],
    "proyectos": ["proyectos", "projects", "portfolio", "portafolio"],
}

#: Secciones sin las cuales el parser de un ATS queda incompleto.
CORE_SECTIONS = ("experiencia", "educacion", "skills")

# --- Verbos de acción -----------------------------------------------------

ACTION_VERBS: Set[str] = {
    # Español
    "lidere", "lideré", "dirigi", "dirigí", "gestione", "gestioné",
    "implemente", "implementé", "desarrolle", "desarrollé", "disene", "diseñé",
    "optimice", "optimicé", "reduje", "aumente", "aumenté", "automatice",
    "automaticé", "migre", "migré", "construi", "construí", "lance", "lancé",
    "coordine", "coordiné", "negocie", "negocié", "escale", "escalé",
    "supervise", "supervisé", "entrene", "entrené", "analice", "analicé",
    "monitorice", "monitoricé", "implante", "implanté", "redacte", "redacté",
    "presente", "presenté", "resolvi", "resolví", "impulse", "impulsé",
    "defini", "definí", "cree", "creé", "integre", "integré", "mejore",
    "mejoré", "consolide", "consolidé", "duplique", "dupliqué",
    # Inglés
    "led", "managed", "built", "designed", "implemented", "developed",
    "launched", "reduced", "increased", "improved", "automated", "migrated",
    "scaled", "owned", "drove", "delivered", "architected", "coordinated",
    "negotiated", "mentored", "trained", "analyzed", "defined", "created",
    "integrated", "shipped", "streamlined", "cut", "grew", "saved",
}

# --- Señales de maquetación problemática ---------------------------------

#: Glifos habituales en plantillas de diseño que muchos parsers pierden.
RISKY_GLYPHS = "▪◦●○■□◆◼➢➤★☆✔✦❖⚫⬤|│┃»◘"

#: Frases típicas de inyección de prompt escondidas en CVs.
INJECTION_PATTERNS: List[str] = [
    r"ignore (all |any |the )?(previous|prior|above) instructions",
    r"ignora (todas )?las instrucciones (previas|anteriores)",
    r"disregard (the )?(previous|prior|above)",
    r"you are (now )?(an? )?(ai|assistant|recruiter)",
    r"(rate|score|puntua|puntúa) (this|me|el cv|este cv).{0,30}(100|10/10|maximum|máxim)",
    r"(recommend|recomienda|selecciona) (this|al|el) (candidate|candidato)",
    r"system prompt",
    r"as an ai language model",
]

# --- Seniority ------------------------------------------------------------

SENIORITY_ORDER: List[str] = [
    "intern", "junior", "mid", "senior", "lead", "manager", "director", "executive",
]

SENIORITY_SIGNALS: Dict[str, List[str]] = {
    "intern": ["intern", "becario", "practicas", "prácticas", "trainee"],
    "junior": ["junior", "jr", "entry level", "nivel inicial"],
    "mid": ["mid", "semi senior", "semi-senior", "ssr", "intermedio"],
    "senior": ["senior", "sr", "especialista", "specialist"],
    "lead": ["lead", "tech lead", "team lead", "principal", "staff", "responsable de equipo"],
    "manager": ["manager", "jefe", "gerente", "head of", "supervisor"],
    "director": ["director", "vp", "vice president"],
    "executive": ["cto", "ceo", "coo", "cfo", "chief", "c-level"],
}

# --- Diccionario de skills (camino sin LLM) ------------------------------

#: Se usa para extraer requisitos de la oferta cuando el LLM está desactivado.
#: Cada entrada es (término canónico, alias).
SKILL_DICTIONARY: Dict[str, List[str]] = {
    # Lenguajes
    "python": ["python3", "py"],
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "java": [],
    "kotlin": [],
    "swift": [],
    "go": ["golang"],
    "rust": [],
    "c++": ["cpp"],
    "c#": ["csharp", "c sharp", ".net"],
    "php": [],
    "ruby": ["ruby on rails", "rails"],
    "scala": [],
    "r": [],
    "sql": [],
    "bash": ["shell scripting", "shell"],
    # Frontend
    "react": ["reactjs", "react.js"],
    "vue": ["vuejs", "vue.js"],
    "angular": ["angularjs"],
    "next.js": ["nextjs"],
    "tailwind": ["tailwindcss"],
    "html": ["html5"],
    "css": ["css3", "sass", "scss"],
    # Backend / datos
    "node.js": ["nodejs", "node"],
    "django": [],
    "flask": [],
    "fastapi": [],
    "spring": ["spring boot"],
    "graphql": [],
    "rest": ["rest api", "api rest", "restful"],
    "postgresql": ["postgres"],
    "mysql": ["mariadb"],
    "mongodb": ["mongo"],
    "redis": [],
    "elasticsearch": ["opensearch"],
    "kafka": ["apache kafka"],
    "spark": ["pyspark", "apache spark"],
    "airflow": ["apache airflow"],
    "dbt": [],
    "snowflake": [],
    "bigquery": [],
    "databricks": [],
    "etl": ["elt"],
    # Cloud / infra
    "aws": ["amazon web services"],
    "azure": ["microsoft azure"],
    "gcp": ["google cloud", "google cloud platform"],
    "docker": [],
    "kubernetes": ["k8s"],
    "terraform": ["iac", "infrastructure as code"],
    "jenkins": [],
    "github actions": ["gh actions"],
    "ci/cd": ["cicd", "ci cd", "integracion continua", "integración continua"],
    "linux": ["unix"],
    "observability": ["observabilidad", "monitoring", "monitorizacion", "monitorización"],
    # Datos / IA
    "machine learning": ["ml", "aprendizaje automatico", "aprendizaje automático"],
    "deep learning": ["redes neuronales", "neural networks"],
    "pytorch": [],
    "tensorflow": [],
    "scikit-learn": ["sklearn"],
    "pandas": [],
    "numpy": [],
    "llm": ["large language model", "modelos de lenguaje", "genai", "ia generativa"],
    "rag": ["retrieval augmented generation"],
    "langchain": [],
    "langgraph": [],
    "nlp": ["procesamiento de lenguaje natural"],
    "computer vision": ["vision por computador", "visión por computador"],
    "power bi": ["powerbi"],
    "tableau": [],
    "looker": [],
    "excel": ["hojas de calculo", "hojas de cálculo"],
    # Producto / gestión
    "scrum": ["agile", "agil", "ágil"],
    "kanban": [],
    "jira": [],
    "confluence": [],
    "product management": ["gestion de producto", "gestión de producto"],
    "stakeholder management": ["gestion de stakeholders", "gestión de stakeholders"],
    "roadmap": [],
    "okr": ["okrs"],
    "a/b testing": ["ab testing", "test a/b"],
    "seo": [],
    "crm": ["salesforce", "hubspot"],
    "erp": ["sap"],
    # Blandas
    "comunicacion": ["comunicación", "communication"],
    "liderazgo": ["leadership"],
    "trabajo en equipo": ["teamwork", "colaboracion", "colaboración"],
    "resolucion de problemas": ["resolución de problemas", "problem solving"],
    "ingles": ["inglés", "english"],
}

SOFT_SKILL_TERMS = {
    "comunicacion", "liderazgo", "trabajo en equipo", "resolucion de problemas",
    "stakeholder management",
}

#: Palabras vacías para la extracción de n-gramas sin LLM.
STOPWORDS: Set[str] = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "en",
    "con", "para", "por", "del", "al", "que", "se", "su", "sus", "lo", "como",
    "mas", "más", "muy", "sobre", "entre", "este", "esta", "estos", "estas",
    "ser", "tener", "hacer", "sera", "será", "tu", "nuestro", "nuestra",
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "at",
    "by", "is", "are", "be", "will", "you", "your", "we", "our", "as", "from",
    "this", "that", "it", "have", "has", "their", "they", "who", "what",
    "experiencia", "experience", "años", "anos", "years", "puesto", "empresa",
    "equipo", "trabajo", "role", "rol", "job", "candidato", "candidate",
    "buscamos", "requisitos", "ofrecemos", "responsabilidades", "requirements",
}
