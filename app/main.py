from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import MatchRequestSchema
from .matcher import MatcherService

app = FastAPI(
    title="Job Matching ML Service",
    description=(
        "Content-Based Job Recommendation menggunakan "
        "Multilingual Sentence Embedding + "
        "Multi-Criteria Weighted Scoring. "
        "Bobot ditentukan via ablation study (NDCG@10 = 0.76963)."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load matcher satu kali saat startup
# (model embedding di-load di sini, bukan per request)
matcher = MatcherService()


@app.get("/")
def root():
    return {
        "service": "ML Job Matching Service",
        "status":  "running",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
def match(payload: MatchRequestSchema):
    """
    Endpoint utama job matching.

    Menerima:
      - pelamar: data lengkap pelamar (skills, pendidikan, pengalaman)
      - lowongans: list loker aktif dari event yang sedang berjalan

    Catatan:
      Filter loker berdasarkan event aktif dilakukan di Laravel
      sebelum request dikirim ke endpoint ini.
      ML service hanya melakukan scoring dan ranking.

    Mengembalikan:
      List loker yang sudah diranking beserta score breakdown
      dan reasoning (tags + reasons) per loker.
    """
    try:
        if not payload.lowongans:
            raise HTTPException(
                status_code=400,
                detail="Tidak ada lowongan yang dikirim."
            )

        result = matcher.match(payload)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )
