from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    MatchRequestSchema,
    RankApplicantsRequestSchema,
    PelamarEmbeddingRequestSchema,
    LowonganEmbeddingRequestSchema,
)
from .embedding import EmbeddingService
from .matcher import MatcherService
from .preprocess import TextPreprocessor
from .ranker import RankerService

import logging
import json
import os

# ============================================================
# LOGGER CONFIG
# ============================================================

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "ml-ranking.log")

logger = logging.getLogger("ml-ranking")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Job Matching ML Service",
    description=(
        "Content-Based Job Recommendation menggunakan "
        "Multilingual Sentence Embedding + "
        "Multi-Criteria Weighted Scoring. "
    ),
    version="2.0.0"
)

# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD SERVICE SEKALI SAAT STARTUP — shared embedding instance
# ============================================================
embedding_service = EmbeddingService()
matcher           = MatcherService(embedding_service)
ranker_service    = RankerService(embedding_service)


# ============================================================
# ROOT
# ============================================================
@app.get("/")
def root():
    return {
        "service": "ML Job Matching Service",
        "status": "running",
        "version": "2.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/detailed")
def detailed_health():
    """Return actionable service diagnostics for the admin health page."""
    try:
        model_loaded = embedding_service.model is not None
        return {
            "status": "ok" if model_loaded else "degraded",
            "model": EmbeddingService.MODEL_NAME,
            "model_loaded": model_loaded,
            "embedding_dimension": 384,
            "scoring_engine": "weighted-linear-v2",
        }
    except Exception as exc:
        logger.exception("HEALTH CHECK DETAIL GAGAL")
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/embeddings/pelamar")
def create_pelamar_embeddings(payload: PelamarEmbeddingRequestSchema):
    """Build canonical applicant texts and return vectors for Laravel to persist."""
    try:
        pelamar = payload.pelamar
        records = [("pelamar_cv", pelamar.id, TextPreprocessor.build_pelamar_text(pelamar))]

        records.extend(
            ("pelamar_skill", skill.id, TextPreprocessor.normalize_text(skill.namaskill))
            for skill in pelamar.skills
            if skill.id is not None and skill.namaskill.strip()
        )

        jurusan = TextPreprocessor.get_jurusan_pelamar(pelamar.pendidikans)
        if jurusan:
            pendidikan = TextPreprocessor._get_pendidikan_tertinggi(pelamar.pendidikans)
            if pendidikan and pendidikan.id is not None:
                records.append((
                    "pelamar_education",
                    pendidikan.id,
                    TextPreprocessor.normalize_text(f"jurusan {jurusan}"),
                ))

        records.extend(
            (
                "pelamar_pengalaman",
                pengalaman.id,
                TextPreprocessor.normalize_text(
                    f"pengalaman kerja sebagai {pengalaman.posisi}"
                ),
            )
            for pengalaman in pelamar.pengalamans
            if pengalaman.id is not None and pengalaman.posisi.strip()
        )

        texts = [text for _, _, text in records]
        vectors = embedding_service.encode_batch(texts)

        return {
            "success": True,
            "model_version": EmbeddingService.MODEL_NAME,
            "embedding_dimension": 384,
            "embeddings": [
                {
                    "embeddable_type": entity_type,
                    "embeddable_id": entity_id,
                    "source_text": source_text,
                    "vector": vector.astype(float).tolist(),
                }
                for (entity_type, entity_id, source_text), vector in zip(records, vectors)
            ],
        }
    except Exception as e:
        logger.exception("ERROR SAAT MEMBUAT EMBEDDING PELAMAR")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/lowongan")
def create_lowongan_embeddings(payload: LowonganEmbeddingRequestSchema):
    """Build canonical vacancy texts and return vectors for Laravel to persist."""
    try:
        lowongan = payload.lowongan
        records = [
            ("lowongan_requirement", lowongan.id, TextPreprocessor.build_lowongan_text(lowongan)),
            ("lowongan_title", lowongan.id, TextPreprocessor.normalize_text(lowongan.namalowongan)),
        ]

        records.extend(
            ("lowongan_skill", skill.id, TextPreprocessor.normalize_text(skill.nama))
            for skill in lowongan.skills
            if skill.nama.strip()
        )
        records.extend(
            (
                "lowongan_education",
                jurusan.id,
                TextPreprocessor.normalize_text(f"jurusan {jurusan.nama}"),
            )
            for jurusan in lowongan.jurusans
            if jurusan.nama.strip()
        )

        texts = [text for _, _, text in records]
        vectors = embedding_service.encode_batch(texts)

        return {
            "success": True,
            "model_version": EmbeddingService.MODEL_NAME,
            "embedding_dimension": 384,
            "embeddings": [
                {
                    "embeddable_type": entity_type,
                    "embeddable_id": entity_id,
                    "source_text": source_text,
                    "vector": vector.astype(float).tolist(),
                }
                for (entity_type, entity_id, source_text), vector in zip(records, vectors)
            ],
        }
    except Exception as e:
        logger.exception("ERROR SAAT MEMBUAT EMBEDDING LOWONGAN")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MATCH JOB ENDPOINT
# ============================================================
@app.post("/match")
def match(payload: MatchRequestSchema):
    """
    Endpoint utama job matching.
    """
    try:
        logger.info(
            "MATCH_REQUEST | pelamar_id=%s | pelamar_nama=%s | lowongan_count=%s | scoring_config=%s",
            getattr(payload.pelamar, 'id', None),
            getattr(payload.pelamar, 'namalengkap', '-'),
            len(payload.lowongans),
            payload.scoring_config.model_dump(),
        )

        if not payload.lowongans:
            raise HTTPException(
                status_code=400,
                detail="Tidak ada lowongan yang dikirim."
            )

        result = matcher.match(payload)

        logger.info(
            "MATCH_RESPONSE | pelamar_id=%s | pelamar_nama=%s | total=%s | top_lowongan_id=%s | top_lowongan_nama=%s | top_final=%.4f | top_label=%s",
            result.get("pelamar_id"),
            result.get("pelamar_nama", "-"),
            result.get("total"),
            result.get("recommendations", [{}])[0].get("lowongan_id", "-") if result.get("recommendations") else "-",
            result.get("recommendations", [{}])[0].get("namalowongan", "-") if result.get("recommendations") else "-",
            result.get("recommendations", [{}])[0].get("final_score", 0.0) if result.get("recommendations") else 0.0,
            result.get("recommendations", [{}])[0].get("label", "-") if result.get("recommendations") else "-",
        )

        return result

    except Exception as e:
        logger.exception("ERROR SAAT MATCHING")

        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


# ============================================================
# RANK APPLICANTS ENDPOINT
# ============================================================
@app.post("/rank-applicants")
async def rank_applicants(
    payload: RankApplicantsRequestSchema
):
    """
    Ranking pelamar yang apply ke satu lowongan.
    """
    try:
        logger.info(
            "RANK_REQUEST | lowongan_id=%s | lowongan=%s | pelamar_count=%s | pelamar_preview=%s | scoring_config=%s",
            getattr(payload.lowongan, 'id', None),
            getattr(payload.lowongan, 'namalowongan', '-'),
            len(payload.pelamars),
            [getattr(p, 'namalengkap', '-') for p in payload.pelamars[:3]],
            payload.scoring_config.model_dump(),
        )

        result = ranker_service.rank(payload)

        ranked = result.get("ranked_applicants", [])

        if ranked:
            top = ranked[0]
            logger.info(
                "RANK_RESPONSE | lowongan_id=%s | lowongan_nama=%s | total=%s | top_rank=%s | top_pelamar_id=%s | top_pelamar_nama=%s | final=%.4f | semantic=%.4f | skill=%.4f | edu=%.4f | exp=%.4f",
                result.get("lowongan_id"),
                result.get("namalowongan", "-"),
                len(ranked),
                top.get("rank"),
                top.get("pelamar_id"),
                top.get("namalengkap", "-"),
                top.get("final_score", 0.0),
                top.get("semantic_score", 0.0),
                top.get("skill_score", 0.0),
                top.get("education_score", 0.0),
                top.get("experience_score", 0.0),
            )
        else:
            logger.info("RANK_RESPONSE | lowongan_id=%s | total=0", result.get("lowongan_id"))

        return result

    except Exception as e:
        logger.exception("ERROR SAAT RANKING APPLICANTS")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )