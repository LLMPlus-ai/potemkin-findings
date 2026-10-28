"""Central configuration: every path, memory cap, and dictionary used by the
numbered scripts lives here so that the pipeline is traceable end to end.

All outputs, temporary files, and DuckDB spill directories stay inside the
project root (PROJECT). Source datasets are read in place, read-only.
"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

# ---- source data (read-only) -------------------------------------------------
SKILLS_DIR = Path("/Volumes/Crucial2/Revelio User Skills")
POSITIONS_DIR = Path("/Volumes/Crucial/Revelio/Positions")
POSTINGS_DIR = Path("/Volumes/Crucial/Revelio/Job Postings")
COMMON_MAP = Path("/Volumes/Crucial/Revelio/revelio_common_mapping/eomys1hnpmzklyyp.parquet")
COMPUSTAT_ANNUAL_DTA = Path("/Volumes/Crucial/Compustat/Compustat North America Fundamentals Annual/"
                            "20250102updated north fundamental_dta/data.dta")
COMPUSTAT_QUARTERLY_CSV = Path("/Volumes/Crucial/Compustat/Compustat North America - Fundamentals Quarterly 196103-202606/"
                               "Compustat North America - Fundamentals Quarterly 196103-202606.csv")
EXECUCOMP_DIR = Path("/Volumes/Crucial2/Execucomp/Execucomp_till_2025")
IBES_DETAIL = Path("/Volumes/Crucial/lseg_ibes_analyst_data/lrfjqb3tryr4dj6d.csv")

# ---- project outputs ---------------------------------------------------------
RAW_EXTRACTS = PROJECT / "data" / "raw_extracts"
EXTERNAL = PROJECT / "data" / "external"
ANALYSIS = PROJECT / "data" / "analysis"
INTERMEDIATE = PROJECT / "intermediate"
TABLES = PROJECT / "tables"
FIGURES = PROJECT / "figures"
LOGS = PROJECT / "logs"
DUCK_TMP = PROJECT / "_scratch" / "duck"

# Memory discipline: the machine has ~10 GB free; every DuckDB connection is
# capped well below that and spills to DUCK_TMP inside the project.
DUCK_MEMORY = "4GB"
DUCK_THREADS = 6


def duck(memory: str = DUCK_MEMORY, threads: int = DUCK_THREADS):
    import duckdb
    DUCK_TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory='{DUCK_TMP}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    return con


# ---- skill dictionaries (applied to the lower-cased, trimmed raw or
# translated skill string; NOT to Revelio's canonical skill_k35000 label,
# whose mapping is audited in scripts/01 and found to be noisy, e.g.
# 97% of rows mapped to "GitHub Copilot" carry the raw string "github") ----
GENAI_EXACT = [
    "generative ai", "genai", "gen ai", "generative ai tools", "generative artificial intelligence",
    "generativeai", "genrative ai", "genarative ai", "generative ai basics", "ai generative",
    "large language models (llm)", "large language models", "large language model",
    "large language model (llm)", "llm", "llms", "large language models (llms)",
    "prompt engineering", "prompt-engineering", "advanced prompt engineering", "prompt design",
    "chatgpt", "chat gpt", "chat-gpt", "gpt-3", "gpt-4", "gpt3", "gpt4", "gpt-4o", "openai api",
    "openai", "github copilot", "git hub copilot", "microsoft copilot", "microsoft copilot studio",
    "copilot studio", "microsoft 365 copilot", "langchain", "llamaindex",
    "retrieval-augmented generation (rag)", "retrieval augmented generation",
    "retrieval-augmented generation", "midjourney", "stable diffusion", "dall-e", "dalle",
    "ai agents", "agentic ai", "google gemini", "claude ai", "anthropic claude",
]
# Core (pre-generative) AI / machine-learning skills
AICORE_EXACT = [
    "machine learning", "deep learning", "artificial intelligence", "artificial intelligence (ai)",
    "neural networks", "artificial neural networks", "deep neural networks (dnn)",
    "convolutional neural networks", "convolutional neural networks (cnn)",
    "recurrent neural networks", "natural language processing", "natural language processing (nlp)",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "scikitlearn", "sklearn",
    "reinforcement learning", "supervised learning", "unsupervised learning", "transfer learning",
    "applied machine learning", "machine learning algorithms", "mlops", "mlflow", "kubeflow",
    "aws sagemaker", "amazon sagemaker", "azure machine learning", "xgboost", "opencv",
    "image recognition", "speech recognition", "pattern recognition", "bert",
    "hugging face", "hugging face transformers",
]
