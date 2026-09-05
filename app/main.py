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

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "logs/ml-ranking.log",
            encoding="utf-8"
        )
    ]
)

logger = logging.getLogger("ml-ranking")
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
        logger.info("========== /match REQUEST ==========")

        payload_dict = payload.model_dump()

        logger.info(
            "Payload:\n%s",
            json.dumps(
                payload_dict,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        )

        logger.info(
            "Jumlah lowongan: %s",
            len(payload_dict.get("lowongans", []))
        )

        if not payload.lowongans:
            raise HTTPException(
                status_code=400,
                detail="Tidak ada lowongan yang dikirim."
            )

        result = matcher.match(payload)

        logger.info("========== /match RESPONSE ==========")

        logger.info(
            "Response:\n%s",
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        )

        logger.info("========== /match SELESAI ==========")

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
            "========== /rank-applicants REQUEST =========="
        )

        payload_dict = payload.model_dump()

        logger.info(
            "Payload:\n%s",
            json.dumps(
                payload_dict,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        )

        logger.info(
            "Lowongan: %s",
            payload_dict
            .get("lowongan", {})
            .get("namalowongan", "Unknown")
        )

        logger.info(
            "Lowongan ID: %s",
            payload_dict
            .get("lowongan", {})
            .get("id", "-")
        )

        logger.info(
            "Jumlah pelamar: %s",
            len(payload_dict.get("pelamars", []))
        )

        result = ranker_service.rank(payload)

        logger.info(
            "========== /rank-applicants RESPONSE =========="
        )

        logger.info(
            "Response:\n%s",
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        )

        ranked = result.get("ranked_applicants", [])

        logger.info(
            "Total hasil ranking: %s",
            len(ranked)
        )

        if ranked:
            logger.info("========== TOP 10 ==========")

            for item in ranked[:10]:
                logger.info(
                    (
                        "Rank #%s | "
                        "Pelamar ID=%s | "
                        "Nama=%s | "
                        "Score=%.4f | "
                        "Semantic=%.4f | "
                        "Skill=%.4f | "
                        "Edu=%.4f | "
                        "Exp=%.4f"
                    ),
                    item.get("rank"),
                    item.get("pelamar_id"),
                    item.get("namalengkap"),
                    item.get("final_score", 0),
                    item.get("semantic_score", 0),
                    item.get("skill_score", 0),
                    item.get("education_score", 0),
                    item.get("experience_score", 0)
                )

        logger.info("========== REQUEST SELESAI ==========")

        return result

    except Exception as e:
        logger.exception("ERROR SAAT RANKING APPLICANTS")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )