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
import time
import uuid

# ============================================================
# LOGGER CONFIG
# ============================================================

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        # logging.StreamHandler(),
        logging.FileHandler(
            "logs/ml-ranking.log",
            encoding="utf-8"
        )
    ]
)

logger = logging.getLogger("ml-ranking")


def log_request_metrics(endpoint: str, request_type: str, started_at, **extra):
    """Write a compact, structured performance record for each request."""
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    payload = {
        "request_id": extra.get("request_id") or uuid.uuid4().hex,
        "request_type": request_type,
        "endpoint": endpoint,
        "duration_ms": duration_ms,
        "status": extra.get("status", "success"),
        "model_version": extra.get("model_version"),
        "embedding_dimension": extra.get("embedding_dimension"),
        "total_records": extra.get("total_records"),
        "pelamar_id": extra.get("pelamar_id"),
        "lowongan_id": extra.get("lowongan_id"),
        "records_count": extra.get("records_count"),
        "extra": extra.get("extra") or {},
    }
    logger.info("REQUEST_METRICS %s", json.dumps(payload, ensure_ascii=False, default=str))
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
    started_at = time.perf_counter()
    request_id = uuid.uuid4().hex
    try:
        pelamar = payload.pelamar

        # raw text mentah sebelum diproses, buat pembanding di TA
        raw_records = [("pelamar_cv", pelamar.id, pelamar.deskripsidiri or "")]

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

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        result = {
            "success": True,
            "request_id": request_id,
            "request_type": "embedding_pelamar",
            "model_version": EmbeddingService.MODEL_NAME,
            "embedding_dimension": 384,
            "processing_time_ms": elapsed_ms,
            "total_records": len(records),
            "embeddings": [
                {
                    "embeddable_type": entity_type,
                    "embeddable_id": entity_id,
                    "source_text": source_text,
                    "vector_preview": vector.astype(float).tolist()[:5],  # 5 nilai pertama aja buat preview
                    "vector": vector.astype(float).tolist(),
                }
                for (entity_type, entity_id, source_text), vector in zip(records, vectors)
            ],
        }
        log_request_metrics(
            "/embeddings/pelamar",
            "embedding_pelamar",
            started_at,
            request_id=request_id,
            status="success",
            model_version=EmbeddingService.MODEL_NAME,
            embedding_dimension=384,
            total_records=len(records),
            pelamar_id=pelamar.id,
            extra={
                "skills_count": len(pelamar.skills),
                "pendidikans_count": len(pelamar.pendidikans),
                "pengalamans_count": len(pelamar.pengalamans),
                "is_update": pelamar.id is not None,
            },
        )
        return result
    except Exception as e:
        log_request_metrics(
            "/embeddings/pelamar",
            "embedding_pelamar",
            started_at,
            request_id=request_id,
            status="error",
            pelamar_id=getattr(payload, "pelamar", None).id if getattr(payload, "pelamar", None) else None,
            extra={"error": str(e)},
        )
        logger.exception("ERROR SAAT MEMBUAT EMBEDDING PELAMAR")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings/lowongan")
def create_lowongan_embeddings(payload: LowonganEmbeddingRequestSchema):
    """Build canonical vacancy texts and return vectors for Laravel to persist."""
    started_at = time.perf_counter()
    request_id = uuid.uuid4().hex
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

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        result = {
            "success": True,
            "request_id": request_id,
            "request_type": "embedding_lowongan",
            "model_version": EmbeddingService.MODEL_NAME,
            "embedding_dimension": 384,
            "processing_time_ms": elapsed_ms,
            "total_records": len(records),
            "embeddings": [
                {
                    "embeddable_type": entity_type,
                    "embeddable_id": entity_id,
                    "source_text": source_text,
                    "vector_preview": vector.astype(float).tolist()[:5],
                    "vector": vector.astype(float).tolist(),
                }
                for (entity_type, entity_id, source_text), vector in zip(records, vectors)
            ],
        }
        log_request_metrics(
            "/embeddings/lowongan",
            "embedding_lowongan",
            started_at,
            request_id=request_id,
            status="success",
            model_version=EmbeddingService.MODEL_NAME,
            embedding_dimension=384,
            total_records=len(records),
            lowongan_id=lowongan.id,
            extra={
                "skills_count": len(lowongan.skills),
                "jurusans_count": len(lowongan.jurusans),
                "is_update": lowongan.id is not None,
            },
        )
        return result
    except Exception as e:
        log_request_metrics(
            "/embeddings/lowongan",
            "embedding_lowongan",
            started_at,
            request_id=request_id,
            status="error",
            lowongan_id=getattr(payload, "lowongan", None).id if getattr(payload, "lowongan", None) else None,
            extra={"error": str(e)},
        )
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
    started_at = time.perf_counter()
    request_id = uuid.uuid4().hex
    try:
        logger.info("========== /match REQUEST ==========")

        payload_dict = payload.model_dump()
        payload_for_log = truncate_embeddings_for_log(payload_dict)

        logger.info(
            "Payload:\n%s",
            json.dumps(
                payload_for_log,
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

        result["request_id"] = request_id
        result["request_type"] = "match"
        if "processing_time_ms" not in result:
            result["processing_time_ms"] = round((time.perf_counter() - started_at) * 1000, 2)

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
        log_request_metrics(
            "/match",
            "match",
            started_at,
            request_id=request_id,
            status="success",
            total_records=len(payload.lowongans or []),
            pelamar_id=getattr(payload.pelamar, "id", None),
            extra={
                "total_lowongans_sent": len(payload.lowongans or []),
                "total_recommendations": len(result.get("recommendations", [])),
            },
        )

        return result

    except Exception as e:
        log_request_metrics(
            "/match",
            "match",
            started_at,
            request_id=request_id,
            status="error",
            total_records=len(payload.lowongans or []),
            extra={"error": str(e)},
        )
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
    started_at = time.perf_counter()
    request_id = uuid.uuid4().hex
    try:
        logger.info(
            "========== /rank-applicants REQUEST =========="
        )

        payload_dict = payload.model_dump()
        payload_for_log = truncate_embeddings_for_log(payload_dict)

        logger.info(
            "Payload:\n%s",
            json.dumps(
                payload_for_log,
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
        log_request_metrics(
            "/rank-applicants",
            "rank_applicants",
            started_at,
            request_id=request_id,
            status="success",
            total_records=len(payload.pelamars or []),
            lowongan_id=getattr(payload.lowongan, "id", None),
            extra={
                "total_pelamars_sent": len(payload.pelamars or []),
                "total_ranked": len(ranked),
            },
        )

        return result

    except Exception as e:
        log_request_metrics(
            "/rank-applicants",
            "rank_applicants",
            started_at,
            request_id=request_id,
            status="error",
            total_records=len(payload.pelamars or []),
            lowongan_id=getattr(payload.lowongan, "id", None),
            extra={"error": str(e)},
        )
        logger.exception("ERROR SAAT RANKING APPLICANTS")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        
def truncate_embeddings_for_log(data, max_len: int = 5):
    """Recursively truncate any field whose key contains 'embedding'
    so logs stay readable instead of dumping 384 floats."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if "embedding" in key.lower() and isinstance(value, list):
                result[key] = {
                    "preview": value[:max_len],
                    "total_dim": len(value),
                }
            else:
                result[key] = truncate_embeddings_for_log(value, max_len)
        return result
    elif isinstance(data, list):
        return [truncate_embeddings_for_log(item, max_len) for item in data]
    else:
        return data      
