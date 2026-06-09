import math
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

CURRENT_YEAR = datetime.now().year

# BOBOT WEIGHTED SCORING
W_SEMANTIC = 0.50
W_SKILL    = 0.10
W_EDU      = 0.15
W_EXP      = 0.25

# THRESHOLD COSINE UNTUK SKILL MATCH
SKILL_THRESHOLD = 0.35

# PROFICIENCY WEIGHT
PROFICIENCY_WEIGHT = {
    'Kurang':      0.25,
    'Cukup':       0.50,
    'Baik':        0.75,
    'Sangat Baik': 1.00,
}

# EDU LEVEL ORDINAL
def _edu_level_from_str(kategori: str) -> int:
    k = str(kategori).upper().strip()
    if 'S3'  in k: return 10
    if 'S2'  in k: return 8
    if 'S1'  in k: return 6
    if 'D4'  in k: return 6
    if 'D3'  in k: return 5
    if 'D2'  in k: return 4
    if 'D1'  in k: return 4
    if 'SMK' in k: return 3
    if 'SMA' in k: return 3
    if 'SMP' in k: return 2
    if 'SD'  in k: return 1
    return 0

# Lambda recency decay  λ=0.15: exp 5thn lalu = e^(-0.75) = 47%, exp 10thn lalu = 22%
RECENCY_LAMBDA = 0.15

class ScoringService:
    # S1 — SEMANTIC SCORE
    @staticmethod
    def semantic_similarity(
        cv_vec: np.ndarray,
        job_vec: np.ndarray
    ) -> float:
        if cv_vec is None or job_vec is None:
            return 0.0
        score = cosine_similarity([cv_vec], [job_vec])[0][0]
        return round(float(score), 4)

    # S2 — SKILL SCORE
    @staticmethod
    def skill_score(
        skills,
        job_vec: np.ndarray,
        embedding_service,
        threshold: float = SKILL_THRESHOLD
    ) -> float:
        if not skills or job_vec is None:
            return 0.0

        total_w   = 0.0
        matched_w = 0.0

        for skill in skills:
            w = PROFICIENCY_WEIGHT.get(skill.keterangan, 0.5)
            total_w += w

            # Encode dalam konteks kalimat untuk representasi lebih baik
            skill_text = f"memiliki keahlian {skill.namaskill}"
            skill_vec  = embedding_service.encode(skill_text)

            similarity = float(
                cosine_similarity([skill_vec], [job_vec])[0][0]
            )

            if similarity >= threshold:
                matched_w += w

        return round(matched_w / total_w, 4) if total_w > 0 else 0.0

    # S3 — EDUCATION SCORE
    @staticmethod
    def education_score(
        pendidikans,
        job_vec: np.ndarray,
        embedding_service
    ) -> float:
        if not pendidikans or job_vec is None:
            return 0.2

        # Ambil pendidikan tertinggi
        def level(p):
            return _edu_level_from_str(p.kategori)

        edu_tertinggi = max(pendidikans, key=level)

        # Dimensi 1: jenjang ordinal
        level_int   = _edu_level_from_str(edu_tertinggi.kategori)
        level_score = level_int / 10.0

        # Dimensi 2: relevansi jurusan (semantic)
        if edu_tertinggi.jurusan and edu_tertinggi.jurusan.strip():
            jurusan_text  = (
                f"latar belakang pendidikan jurusan "
                f"{edu_tertinggi.jurusan}"
            )
            jurusan_vec   = embedding_service.encode(jurusan_text)
            jurusan_score = float(
                cosine_similarity([jurusan_vec], [job_vec])[0][0]
            )
        else:
            jurusan_score = 0.3  # tidak ada jurusan = netral rendah

        return round(0.35 * level_score + 0.65 * jurusan_score, 4)

    # S4 — EXPERIENCE SCORE
    @staticmethod
    def experience_score(
        pengalamans,
        job_vec: np.ndarray,
        embedding_service
    ) -> float:
        if not pengalamans or job_vec is None:
            return 0.1  # tidak ada pengalaman = sangat rendah

        scores = []

        for exp in pengalamans:
            thn_awal  = int(exp.tahunawal)
            thn_akhir = (
                CURRENT_YEAR
                if exp.aktif == 1 or exp.tahunselesai is None
                else int(exp.tahunselesai)
            )
            thn_akhir = max(thn_awal, min(thn_akhir, CURRENT_YEAR))

            # Relevance — encode posisi dalam konteks kalimat
            posisi_text = f"pengalaman kerja sebagai {exp.posisi}"
            posisi_vec  = embedding_service.encode(posisi_text)
            relevance   = float(
                cosine_similarity([posisi_vec], [job_vec])[0][0]
            )

            # Duration weight (cap 5 tahun)
            durasi = max(thn_akhir - thn_awal, 0)
            dur_w  = min(durasi / 5.0, 1.0)

            # Recency weight (exponential decay)
            berlalu = max(CURRENT_YEAR - thn_akhir, 0)
            rec_w   = math.exp(-RECENCY_LAMBDA * berlalu)

            score_i = relevance * (0.6 + 0.2 * dur_w + 0.2 * rec_w)
            scores.append(score_i)

        return round(float(max(scores)), 4) if scores else 0.1

    # FINAL SCORE — Weighted Linear Combination
    @staticmethod
    def final_score(
        semantic: float,
        skill: float,
        edu: float,
        exp: float
    ) -> float:
        score = (
            W_SEMANTIC * semantic +
            W_SKILL    * skill    +
            W_EDU      * edu      +
            W_EXP      * exp
        )
        return round(score, 4)

    # CLASSIFY — Label kecocokan
    @staticmethod
    def classify(score: float) -> str:
        if score >= 0.55:
            return 'Sangat Cocok'
        if score >= 0.40:
            return 'Cukup Cocok'
        return 'Kurang Cocok'

    # COLOR & PERCENTAGE
    @staticmethod
    def determine_color(score: float) -> str:
        if score >= 0.55:
            return 'green'
        if score >= 0.40:
            return 'yellow'
        return 'red'

    @staticmethod
    def percentage(score: float) -> int:
        return round(score * 100)
