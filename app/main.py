from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    MatchRequestSchema,
    RankApplicantsRequestSchema
)
from .matcher import MatcherService
from .ranker import RankerService

import logging
import json

# ============================================================
# LOGGER CONFIG
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
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
        "Bobot ditentukan via ablation study "
        "(NDCG@10 = 0.76963)."
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
# LOAD SERVICE SEKALI SAAT STARTUP
# ============================================================
matcher = MatcherService()
ranker_service = RankerService()


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

        # ====================================================
        # LOG PAYLOAD MASUK DARI LARAVEL
        # ====================================================
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

        # ====================================================
        # PROSES RANKING
        # ====================================================
        result = ranker_service.rank(payload)

        # ====================================================
        # LOG RESPONSE
        # ====================================================
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

        # Ringkasan top ranking
        ranked = result.get(
            "ranked_applicants",
            []
        )

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

        logger.info(
            "========== REQUEST SELESAI =========="
        )

        return result

    except Exception as e:
        logger.exception(
            "ERROR SAAT RANKING APPLICANTS"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )