# Incident_Analysis.py
# Hybrid RAG — combines curated regulations (rag_knowledge_base.py)
# with a PDF-derived FAISS index (faiss_construction_index/)
# If PDF index is missing, falls back to curated regulations only — no crash

import os
import logging
import pickle
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class SafetyRAGEngine:
    """
    Hybrid RAG engine with two retrieval sources:

    SOURCE A — Curated regulations (rag_knowledge_base.py)
        30 hand-picked OSHA + Indian construction regulations
        Zero noise, always available, instant load
        Own FAISS index built in-memory at startup

    SOURCE B — PDF-derived FAISS index (faiss_construction_index/)
        Built from actual legal PDFs on Kaggle
        Loaded from disk if available
        Gracefully skipped if folder is missing

    Retrieval strategy:
        Query both sources → merge → deduplicate → return top-k
    """

    def __init__(self):
        self._embedder        = None
        self._curated_index   = None
        self._curated_regs    = None
        self._pdf_index       = None
        self._pdf_docs        = None
        self._initialized     = False

    def _lazy_init(self):
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
            from rag_knowledge_base import REGULATIONS

            logger.info("RAG: Loading sentence-transformer (all-MiniLM-L6-v2)...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # SOURCE A: build curated index in-memory
            logger.info(f"RAG: Embedding {len(REGULATIONS)} curated regulations...")
            self._curated_regs = REGULATIONS
            curated_embeddings = self._embedder.encode(
                REGULATIONS,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype("float32")

            dim = curated_embeddings.shape[1]
            self._curated_index = faiss.IndexFlatIP(dim)
            self._curated_index.add(curated_embeddings)
            logger.info(f"RAG: Source A ready — {self._curated_index.ntotal} curated regulations")

            # SOURCE B: load PDF-derived index from disk (FIXED WIN ESCAPE VIA RAW STRING 'r')
            index_dir  = r"C:\Users\Anuj\Documents\AI_Safety_System\faiss_construction_index"
            faiss_path = os.path.join(index_dir, "index.faiss")
            pkl_path   = os.path.join(index_dir, "index.pkl")

            if os.path.exists(faiss_path) and os.path.exists(pkl_path):
                logger.info(f"RAG: Loading PDF index from {index_dir}...")
                self._pdf_index = faiss.read_index(faiss_path)

                with open(pkl_path, "rb") as f:
                    vectorstore_data = pickle.load(f)

                # NEW FIXED CODE
                # Robust unpacking — handles all LangChain FAISS pickle formats
                try:
                    if isinstance(vectorstore_data, tuple):
                        # Format: (InMemoryDocstore, {index: doc_id})
                        docstore = vectorstore_data[0]
                        if hasattr(docstore, "_dict"):
                            # LangChain InMemoryDocstore
                            self._pdf_docs = [
                                doc.page_content
                                for doc in docstore._dict.values()
                            ]
                        elif isinstance(docstore, dict):
                            self._pdf_docs = [
                                doc.page_content
                                for doc in docstore.values()
                            ]
                        else:
                            # Last resort — iterate directly
                            self._pdf_docs = [
                                doc.page_content
                                for doc in list(docstore._dict.values())
                            ]
                    elif hasattr(vectorstore_data, "docstore"):
                        self._pdf_docs = [
                            doc.page_content
                            for doc in vectorstore_data.docstore._dict.values()
                        ]
                    elif isinstance(vectorstore_data, dict):
                        self._pdf_docs = [
                            doc.page_content
                            for doc in vectorstore_data.values()
                        ]
                    else:
                        self._pdf_docs = [
                            doc.page_content for doc in vectorstore_data
                        ]
                    logger.info(f"RAG: Source B ready — {len(self._pdf_docs)} PDF chunks extracted")
                except Exception as pkl_err:
                    logger.error(f"RAG: Cannot unpack pickle: {pkl_err}")
                    logger.warning("RAG: Disabling Source B — curated regulations still active")
                    self._pdf_index = None
                    self._pdf_docs  = None

                logger.info(f"RAG: Source B ready — {self._pdf_index.ntotal} PDF chunks loaded")
            else:
                logger.warning(
                    f"RAG: PDF index not found at '{index_dir}/' — "
                    "using curated regulations only. "
                    "Run construction_rag_builder.py on Kaggle to enable Source B."
                )
                self._pdf_index = None
                self._pdf_docs  = None

            self._initialized = True

        except ImportError as e:
            logger.error(f"RAG: Missing dependency — {e}")
            logger.error("Run: pip install faiss-cpu sentence-transformers")
            self._initialized = False
        except Exception as e:
            logger.error(f"RAG: Init failed — {e}", exc_info=True)
            self._initialized = True   # ← CRITICAL: stop retry loop
            # Source A may still be available even if Source B failed
            # We mark initialized=True so we don't reload the model repeatedly

    def retrieve(self, query: str, k: int = 3) -> list:
        self._lazy_init()

        if not self._initialized:
            return []

        try:
            import numpy as np
            from rag_knowledge_base import VIOLATION_CATEGORY_MAP

            category_terms = VIOLATION_CATEGORY_MAP.get(query, [])
            enriched_query = f"construction safety PPE {query} {' '.join(category_terms)}"

            query_vec = self._embedder.encode(
                [enriched_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")

            results = []

            # Source A (Curated Index Execution)
            scores_a, idx_a = self._curated_index.search(query_vec, k)
            for score, idx in zip(scores_a[0], idx_a[0]):
                if idx != -1 and score > 0.20:
                    # Apply proportional scalar adjustment rather than flat float additions
                    results.append((float(score) * 1.02, self._curated_regs[idx]))

            # Source B (PDF File Vector Store Execution)
            if self._pdf_index is not None and self._pdf_docs:
                scores_b, idx_b = self._pdf_index.search(query_vec, k)
                for score, idx in zip(scores_b[0], idx_b[0]):
                    if idx != -1 and idx < len(self._pdf_docs) and score > 0.20:
                        results.append((float(score), self._pdf_docs[idx]))

            # Merge + deduplicate
            seen   = set()
            unique = []
            for score, text in sorted(results, key=lambda x: x[0], reverse=True):
                key = text[:120].strip().lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(text)
                if len(unique) >= k:
                    break

            sources_used = "A+B" if self._pdf_index else "A only"
            logger.debug(f"RAG: '{query}' → {len(unique)} results (sources: {sources_used})")
            return unique

        except Exception as e:
            logger.error(f"RAG: Retrieval error for '{query}': {e}", exc_info=True)
            return []

    @property
    def status(self) -> dict:
        self._lazy_init()
        return {
            "initialized":      self._initialized,
            "curated_regs":     len(self._curated_regs) if self._curated_regs else 0,
            "pdf_index_loaded": self._pdf_index is not None,
            "pdf_chunks":       self._pdf_index.ntotal if self._pdf_index else 0,
            "total_searchable": (
                (self._curated_index.ntotal if self._curated_index else 0) +
                (self._pdf_index.ntotal if self._pdf_index else 0)
            )
        }


# Single shared instance
_rag_engine = SafetyRAGEngine()


def GROQ_report(incident: dict) -> str:
    API_KEY = os.environ.get("GROQ_API_AI_SAFETY_REPORT")
    if not API_KEY:
        logger.error("GROQ_API_AI_SAFETY_REPORT not set in .env")
        return "Analysis unavailable: API key not configured."

    violation_type   = incident.get("violation_type", "UNKNOWN VIOLATION")
    person_id        = incident.get("person_id", "Unknown")
    duration_seconds = float(incident.get("duration_seconds", 0))
    risk_score       = incident.get("risk_score", 0)
    risk_level       = incident.get("risk_level", "LOW")
    severity         = incident.get("severity", "LOW")
    near_machinery   = incident.get("near_machinery", False)
    confidence       = float(incident.get("confidence", 0))

    # STEP 1: Hybrid RAG retrieval
    retrieved = _rag_engine.retrieve(violation_type, k=3)

    if retrieved:
        reg_context = "\n\nRELEVANT REGULATIONS RETRIEVED:\n"
        for i, reg in enumerate(retrieved, 1):
            reg_text     = reg[:400] if len(reg) > 400 else reg
            reg_context += f"\n[REG-{i}] {reg_text}\n"
        logger.info(f"RAG: {len(retrieved)} regulations injected for '{violation_type}'")
    else:
        reg_context = "\n\n[RAG UNAVAILABLE — no regulations retrieved]\n"
        logger.warning(f"RAG: No results for '{violation_type}'")

    # STEP 2: Prompt
    prompt = f"""You are a certified construction site safety officer and legal compliance expert.

INCIDENT DETAILS:
- Violation     : {violation_type}
- Worker ID     : Person-{person_id}
- Duration      : {duration_seconds:.1f} seconds
- Risk Score    : {risk_score}
- Risk Level    : {risk_level}
- Severity      : {severity}
- Near Machinery: {"YES — ELEVATED DANGER" if near_machinery else "No"}
- Confidence    : {confidence:.1%}
{reg_context}

Using ONLY the retrieved regulations above, respond in EXACTLY this format:

VIOLATION SUMMARY:
[One sentence: what was observed, who, how long]

REGULATORY BREACH:
[Exact regulation number and requirement violated , explain its consequences]

RISK ASSESSMENT:
[Specific danger this poses, mention machinery if relevant]

REQUIRED ACTION:
[What the site supervisor must do immediately]

PENALTY EXPOSURE:
[Exact fine amount from the cited regulation]

Keep each section to 3-4 sentences. Total under 200 words."""

    # STEP 3: Groq LLM
    try:
        client   = Groq(api_key=API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        analysis = response.choices[0].message.content

        # Audit trail
        if retrieved:
            refs = []
            for reg in retrieved:
                if "OSHA" in reg:
                    refs.append(reg.split(":")[0].strip())
                elif reg.startswith("IS "):
                    refs.append(reg.split(":")[0].strip())
                elif "Building and Other" in reg:
                    refs.append("BOCWA 1996")
                elif "National Building Code" in reg:
                    refs.append("NBC India 2016")
                elif "NIOSH" in reg:
                    refs.append("NIOSH Guidelines")
            refs = list(dict.fromkeys(refs))
            if refs:
                analysis += f"\n\n[Regulations consulted: {', '.join(refs)}]"

            status = _rag_engine.status
            if status["pdf_index_loaded"]:
                analysis += f" [Sources: Curated + PDF ({status['pdf_chunks']} chunks)]"
            else:
                analysis += " [Source: Curated regulations]"

        logger.info(f"GROQ: Report generated | {violation_type} | Person-{person_id}")
        return analysis

    except Exception as e:
        logger.error(f"GROQ: API call failed: {e}", exc_info=True)
        fallback = (
            f"VIOLATION SUMMARY: {violation_type} detected for "
            f"Person-{person_id} lasting {duration_seconds:.0f} seconds.\n\n"
            f"RISK ASSESSMENT: {risk_level} risk. Score: {risk_score}."
        )
        if retrieved:
            fallback += f"\n\nRELEVANT REGULATION: {retrieved[0][:300]}"
        return fallback


def get_rag_status() -> dict:
    return _rag_engine.status