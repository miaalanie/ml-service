import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import TextPreprocessor

CURRENT_YEAR = datetime.now().year

# BOBOT WEIGHTED SCORING
W_SEMANTIC = 0.50
W_SKILL    = 0.10
W_EDU      = 0.15
W_EXP      = 0.25

# THRESHOLD COSINE UNTUK SKILL MATCH
SKILL_THRESHOLD = 0.50  # sesuai definisi: >= 0.50


class ScoringService:

    # ============================================================
    # S1 — SEMANTIC SCORE
    # Kemiripan teks profil pelamar vs teks loker secara keseluruhan
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
    # Dari semua skill yang DIMINTA loker, berapa persen ter-cover
    # skill_score = total bobot skill match / total skill diminta loker
    # Match = cosine skill pelamar vs nama skill loker >= 0.50
    # Bobot: Kurang=0.25, Cukup=0.50, Baik=0.75, Sangat Baik=1.00
    # ============================================================
    @staticmethod
    def skill_score(
        pelamar_skills,
        lowongan_skills,
        embedding_service,
        threshold: float = SKILL_THRESHOLD
    ) -> float:
        # Tidak ada skill yang diminta → tidak bisa dinilai → 0
        if not lowongan_skills:
            return 0.0

        # Tidak ada skill pelamar → tidak ada yang match → 0
        if not pelamar_skills:
            return 0.0

        # Pre-encode semua skill pelamar sekali
        pelamar_vecs = []
        for s in pelamar_skills:
            vec = embedding_service.encode(
                TextPreprocessor.normalize_text(s.namaskill)
            )
            pelamar_vecs.append((s, vec))

        total_skill   = len(lowongan_skills)   # tiap skill loker bobot = 1
        matched_score = 0.0

        for loker_skill in lowongan_skills:
            loker_vec = embedding_service.encode(
                TextPreprocessor.normalize_text(loker_skill.nama)
            )

            # Cari skill pelamar paling mirip dengan skill loker ini
            best_sim    = 0.0
            best_weight = 0.0

            for s, s_vec in pelamar_vecs:
                sim = float(
                    cosine_similarity([s_vec], [loker_vec])[0][0]
                )
                if sim > best_sim:
                    best_sim    = sim
                    best_weight = TextPreprocessor.get_skill_weight(
                        s.keterangan
                    )

            if best_sim >= threshold:
                matched_score += best_weight
            # kalau tidak ada yang >= threshold → kontribusi 0

        return round(matched_score / total_skill, 4)

    # ============================================================
    # S3 — EDUCATION SCORE
    # edu_score = (0.5 x level_score) + (0.5 x jurusan_score)
    #
    # level_score:
    #   - Ada syarat minimal → level pelamar / level minimal (cap 1.0)
    #   - Tidak ada syarat   → level pelamar / 9
    #
    # jurusan_score:
    #   - Ada jurusan loker  → cosine jurusan pelamar vs jurusan loker
    #   - Tidak ada jurusan loker → cosine jurusan pelamar vs teks loker
    #   - Pelamar tidak punya jurusan (SD/SMP) → 0.3 netral
    # ============================================================
    @staticmethod
    def education_score(
        pendidikans,
        lowongan,
        job_vec: np.ndarray,
        embedding_service
    ) -> float:
        if not pendidikans:
            return 0.2

        edu = TextPreprocessor._get_pendidikan_tertinggi(pendidikans)
        if not edu:
            return 0.2

        level_pelamar = TextPreprocessor.get_pendidikan_level(
            edu.kategori
        )

        # --- level_score ---
        if lowongan.minimal_pendidikan and lowongan.minimal_pendidikan.kode:
            level_minimal = TextPreprocessor.get_pendidikan_level_from_kode(
                lowongan.minimal_pendidikan.kode
            )
            level_score = min(level_pelamar / level_minimal, 1.0) \
                if level_minimal > 0 else level_pelamar / 9
        else:
            level_score = level_pelamar / 9

        level_score = round(min(level_score, 1.0), 4)

        # --- jurusan_score ---
        jurusan_pelamar = TextPreprocessor.get_jurusan_pelamar(pendidikans)

        if jurusan_pelamar is None:
            # SD/SMP tidak punya jurusan → netral rendah
            jurusan_score = 0.3
        else:
            jurusan_text = TextPreprocessor.normalize_text(
                f"jurusan {jurusan_pelamar}"
            )
            jurusan_vec = embedding_service.encode(jurusan_text)

            if lowongan.jurusans:
                # Bandingkan dengan jurusan yang diminta loker
                # Ambil skor tertinggi dari semua jurusan loker
                best = 0.0
                for lj in lowongan.jurusans:
                    loker_jurusan_text = TextPreprocessor.normalize_text(
                        f"jurusan {lj.nama}"
                    )
                    loker_jurusan_vec = embedding_service.encode(
                        loker_jurusan_text
                    )
                    sim = float(
                        cosine_similarity(
                            [jurusan_vec], [loker_jurusan_vec]
                        )[0][0]
                    )
                    if sim > best:
                        best = sim
                jurusan_score = round(best, 4)
            else:
                # Loker tidak minta jurusan spesifik →
                # bandingkan dengan teks loker secara umum
                if job_vec is not None:
                    jurusan_score = round(float(
                        cosine_similarity(
                            [jurusan_vec], [job_vec]
                        )[0][0]
                    ), 4)
                else:
                    jurusan_score = 0.5  # netral

        edu_score = (0.5 * level_score) + (0.5 * jurusan_score)
        return round(edu_score, 4)

    # ============================================================
    # S4 — EXPERIENCE SCORE
    # exp_score = (0.5 x posisi_score) + (0.5 x durasi_score)
    #
    # posisi_score  = cosine posisi PALING COCOK pelamar vs nama loker
    # durasi_score:
    #   - Ada syarat minimal (bulan) → total_bulan / minimal (cap 1.0)
    #   - Fresh grad boleh (minimal=0) → 1.0 untuk semua
    # ============================================================
    @staticmethod
    def experience_score(
        pengalamans,
        total_pengalaman_bulan: int,
        lowongan,
        embedding_service
    ) -> float:
        minimal_bulan = lowongan.minimal_pengalaman_bulan or 0

        # --- durasi_score ---
        if minimal_bulan == 0:
            # Fresh grad boleh → semua dapat 1.0
            durasi_score = 1.0
        else:
            durasi_score = min(
                total_pengalaman_bulan / minimal_bulan, 1.0
            )

        durasi_score = round(durasi_score, 4)

        # --- posisi_score ---
        if not pengalamans:
            posisi_score = 0.0
        else:
            loker_text = TextPreprocessor.normalize_text(
                lowongan.namalowongan
            )
            loker_vec = embedding_service.encode(loker_text)

            best_sim = 0.0
            for exp in pengalamans:
                posisi_text = TextPreprocessor.normalize_text(
                    f"pengalaman kerja sebagai {exp.posisi}"
                )
                posisi_vec = embedding_service.encode(posisi_text)
                sim = float(
                    cosine_similarity([posisi_vec], [loker_vec])[0][0]
                )
                if sim > best_sim:
                    best_sim = sim

            posisi_score = round(best_sim, 4)

        exp_score = (0.5 * posisi_score) + (0.5 * durasi_score)
        return round(exp_score, 4)

    # ============================================================
    # FINAL SCORE — Weighted Linear Combination
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
        if score >= 0.80:
            return 'Sangat Cocok'
        if score >= 0.55:
            return 'Cocok'
        if score >= 0.40:
            return 'Cukup Cocok'
        return 'Kurang Cocok'

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