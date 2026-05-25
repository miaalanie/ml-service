import math
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

CURRENT_YEAR = datetime.now().year

# ============================================================
# BOBOT WEIGHTED SCORING
# Sumber: ablation study terhadap data historis lamarans
# career day (971 pelamar, 58 loker, 2000 positive samples)
# Metrik evaluasi: NDCG@10
# NDCG optimal: 0.76963
#
# Catatan limitasi:
# Label dari perilaku melamar (implicit feedback).
# Mengandung selection bias dari konteks career day.
# Referensi: Hu et al. (2008) - Collaborative Filtering
#            for Implicit Feedback Datasets
# ============================================================
W_SEMANTIC = 0.50
W_SKILL    = 0.10
W_EDU      = 0.15
W_EXP      = 0.25

# ============================================================
# THRESHOLD COSINE UNTUK SKILL MATCH
# Nilai 0.35 dipakai di Colab dan menghasilkan distribusi
# yang reasonable (skill score mean 0.376 untuk positif
# vs 0.297 untuk negatif — delta +0.079, paling diskriminatif)
# ============================================================
SKILL_THRESHOLD = 0.35

# ============================================================
# PROFICIENCY WEIGHT
# Ordinal scale: Kurang < Cukup < Baik < Sangat Baik
# Mapping ke [0.25, 0.50, 0.75, 1.00]
# ============================================================
PROFICIENCY_WEIGHT = {
    'Kurang':      0.25,
    'Cukup':       0.50,
    'Baik':        0.75,
    'Sangat Baik': 1.00,
}

# ============================================================
# EDU LEVEL ORDINAL
# Sesuai temuan Colab: DB simpan 'SMA/SMK', 'D4/S1' dll
# Pakai LIKE-based check (str contains), bukan exact match
# ============================================================
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

# Lambda recency decay
# λ=0.15: exp 5thn lalu = e^(-0.75) = 47%, exp 10thn lalu = 22%
# Referensi: Schmidt et al. (1986) — experience-performance
#            relationship plateaus
RECENCY_LAMBDA = 0.15


class ScoringService:

    # ============================================================
    # S1 — SEMANTIC SCORE
    # Cosine similarity antara representasi holistik pelamar
    # dan lowongan dalam ruang vektor 384 dimensi.
    #
    # Landasan cosine vs euclidean:
    # Cosine tidak sensitif terhadap magnitude vektor →
    # fair untuk profil berbeda panjang teks.
    # Referensi: Salton & McGill (1983) — Vector Space Model
    # ============================================================
    @staticmethod
    def semantic_similarity(
        cv_vec: np.ndarray,
        job_vec: np.ndarray
    ) -> float:
        if cv_vec is None or job_vec is None:
            return 0.0
        score = cosine_similarity([cv_vec], [job_vec])[0][0]
        return round(float(score), 4)

    # ============================================================
    # S2 — SKILL SCORE
    # Formula: Σ(w_i × match_i) / Σ(w_i)
    #
    # w_i     = proficiency weight skill ke-i
    # match_i = 1 jika cos_sim(encode(skill_i), job_vec) >= θ
    #
    # Landasan weighted by proficiency:
    # Proficiency adalah ordinal signal valid di data.
    # Pelamar 'Sangat Baik' lebih kompeten dari 'Kurang'
    # meski skill name sama. Mengabaikan ini = info terbuang.
    #
    # Landasan cosine untuk match:
    # Menghindari exact string match yang gagal untuk sinonim
    # ("MS Excel" vs "Microsoft Excel"). Pre-trained model
    # sudah encode relasi ini dari corpus besar.
    #
    # Threshold 0.35: dari observasi distribusi similarity
    # di data real (Colab sample analysis).
    # ============================================================
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
            # Landasan: transformer bekerja lebih baik dengan konteks
            skill_text = f"memiliki keahlian {skill.namaskill}"
            skill_vec  = embedding_service.encode(skill_text)

            similarity = float(
                cosine_similarity([skill_vec], [job_vec])[0][0]
            )

            if similarity >= threshold:
                matched_w += w

        return round(matched_w / total_w, 4) if total_w > 0 else 0.0

    # ============================================================
    # S3 — EDUCATION SCORE
    # Formula: 0.35 × level_score + 0.65 × jurusan_score
    #
    # level_score   = edu_level / 10 (ordinal, normalisasi 0-1)
    # jurusan_score = cos_sim(encode(jurusan), job_vec)
    #
    # Pembagian 35/65:
    # Jurusan lebih diskriminatif dari jenjang.
    # S1 Hukum vs S1 Informatika untuk loker IT =
    # jenjang sama, relevansi sangat berbeda.
    # Referensi: Ritter & Vance (2011) — field of study
    # lebih prediktif dari jenjang untuk technical roles.
    #
    # Hanya pendidikan tertinggi:
    # Mewakili kompetensi akademik terkini.
    # Jenjang bawah sudah 'included' dalam jenjang tertinggi.
    #
    # Temuan dari data real:
    # Delta mean pos-neg = +0.0093 (paling kecil di antara 4 komponen)
    # → edu memang kurang diskriminatif karena loker jarang
    #   menyebut syarat pendidikan eksplisit
    # ============================================================
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

    # ============================================================
    # S4 — EXPERIENCE SCORE
    # Formula per pengalaman i:
    #   score_i = relevance_i × (0.6 + 0.2×dur_i + 0.2×rec_i)
    #
    # relevance_i = cos_sim(encode(posisi_i), job_vec)
    # dur_i       = min(durasi_tahun / 5, 1.0)
    # rec_i       = e^(-λ × tahun_berlalu)
    #
    # Agregasi: max(score_i)
    # Landasan max vs mean:
    # Sufficient condition — 1 pengalaman relevan sudah cukup.
    # Mean tidak fair untuk career changer.
    #
    # Duration cap 5 tahun:
    # Schmidt et al. (1986): marginal return of experience
    # plateaus untuk posisi entry-mid level.
    #
    # Recency decay (λ=0.15):
    # Skill obsolescence — teknologi berubah.
    # Pengalaman lama makin kurang relevan.
    #
    # Limitasi yang diakui:
    # Field 'posisi' sering 1-3 kata saja (tidak ada deskripsi
    # pekerjaan di DB) → signal semantic lemah untuk posisi
    # generik seperti "Staff", "Karyawan", "Magang".
    #
    # Temuan dari data real:
    # Delta pos-neg = +0.0090 (kecil karena limitasi posisi pendek)
    # ============================================================
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

    # ============================================================
    # FINAL SCORE — Weighted Linear Combination
    # Bobot dari ablation study (NDCG@10 = 0.76963)
    # ============================================================
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

    # ============================================================
    # CLASSIFY — Label kecocokan
    # ============================================================
    @staticmethod
    def classify(score: float) -> str:
        if score >= 0.55:
            return 'Sangat Cocok'
        if score >= 0.40:
            return 'Cukup Cocok'
        return 'Kurang Cocok'

    # ============================================================
    # COLOR & PERCENTAGE
    # ============================================================
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
