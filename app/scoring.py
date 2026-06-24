import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import TextPreprocessor

CURRENT_YEAR = datetime.now().year

# BOBOT WEIGHTED SCORING
W_SEMANTIC = 0.25
W_SKILL = 0.25
W_EDU = 0.25
W_EXP = 0.25

# THRESHOLD COSINE UNTUK SKILL MATCH
SKILL_THRESHOLD = 0.50  # sesuai definisi: >= 0.50


class ScoringService:

    # ============================================================
    # S1 — SEMANTIC SCORE
    # Kemiripan teks profil pelamar vs teks loker secara keseluruhan
    # ============================================================
    @staticmethod
    def semantic_similarity(cv_vec: np.ndarray, job_vec: np.ndarray) -> float:
        if cv_vec is None or job_vec is None:
            return 0.0
        score = cosine_similarity([cv_vec], [job_vec])[0][0]
        return round(float(score), 4)

    @staticmethod
    def compute_skill_match(
        pelamar_skills,
        lowongan_skills,
        embedding_service,
        threshold: float = SKILL_THRESHOLD,
    ) -> tuple:
        """
        Hitung skill matching SEKALI — return (score, matched, unmatched).
        Dipakai scoring dan reasoning, tidak perlu komputasi ulang.
        """
        if not lowongan_skills:
            return 0.0, [], []
        if not pelamar_skills:
            return 0.0, [], [s.nama for s in lowongan_skills]

        pelamar_encoded = [
            (s, embedding_service.encode(TextPreprocessor.normalize_text(s.namaskill)))
            for s in pelamar_skills
        ]
        p_matrix = np.vstack([v for _, v in pelamar_encoded])

        total_skill   = len(lowongan_skills)
        matched_score = 0.0
        matched       = []
        unmatched     = []

        for loker_skill in lowongan_skills:
            loker_vec = embedding_service.encode(
                TextPreprocessor.normalize_text(loker_skill.nama)
            )
            sims     = cosine_similarity(p_matrix, [loker_vec]).flatten()
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_s   = pelamar_encoded[best_idx][0]

            if best_sim >= threshold:
                matched_score += TextPreprocessor.get_skill_weight(best_s.keterangan)
                matched.append((loker_skill.nama, best_s.namaskill, best_s.keterangan))
            else:
                unmatched.append(loker_skill.nama)

        return round(matched_score / total_skill, 4), matched, unmatched

    @staticmethod
    def compute_exp_for_reasoning(pengalamans, job_vec, embedding_service) -> tuple:
        """
        Cari pengalaman paling relevan vs job_vec untuk reasoning.
        Returns (best_exp, best_sim). Dipakai generate_tags DAN
        generate_reasons — hitung sekali, pass hasilnya.
        """
        if not pengalamans:
            return None, 0.0

        posisi_vecs = [
            embedding_service.encode(
                TextPreprocessor.normalize_text(f"pengalaman kerja sebagai {exp.posisi}")
            )
            for exp in pengalamans
        ]
        p_matrix = np.vstack(posisi_vecs)
        sims     = cosine_similarity(p_matrix, [job_vec]).flatten()
        best_idx = int(np.argmax(sims))
        return pengalamans[best_idx], round(float(sims[best_idx]), 4)

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
        threshold: float = SKILL_THRESHOLD,
    ) -> float:
        score, _, _ = ScoringService.compute_skill_match(
            pelamar_skills, lowongan_skills, embedding_service, threshold
        )
        return score

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
        pendidikans, lowongan, job_vec: np.ndarray, embedding_service
    ) -> float:
        if not pendidikans:
            return 0.2

        edu = TextPreprocessor._get_pendidikan_tertinggi(pendidikans)
        if not edu:
            return 0.2

        level_pelamar = TextPreprocessor.get_pendidikan_level(edu.kategori)

        # --- level_score ---
        if lowongan.minimal_pendidikan and lowongan.minimal_pendidikan.kode:
            level_minimal = TextPreprocessor.get_pendidikan_level_from_kode(
                lowongan.minimal_pendidikan.kode
            )
            level_score = (
                min(level_pelamar / level_minimal, 1.0)
                if level_minimal > 0
                else level_pelamar / 9
            )
        else:
            level_score = level_pelamar / 9

        level_score = round(min(level_score, 1.0), 4)

        # --- jurusan_score ---
        jurusan_pelamar = TextPreprocessor.get_jurusan_pelamar(pendidikans)

        if jurusan_pelamar is None:
            # SD/SMP tidak punya jurusan → netral rendah
            jurusan_score = 0.3
        else:
            jurusan_text = TextPreprocessor.normalize_text(f"jurusan {jurusan_pelamar}")
            jurusan_vec = embedding_service.encode(jurusan_text)

            if lowongan.jurusans:
                # Bandingkan dengan jurusan yang diminta loker
                # Ambil skor tertinggi dari semua jurusan loker
                loker_jurusan_vecs = [
                    embedding_service.encode(
                        TextPreprocessor.normalize_text(f"jurusan {lj.nama}")
                    )
                    for lj in lowongan.jurusans
                ]
                l_matrix      = np.vstack(loker_jurusan_vecs)
                sims          = cosine_similarity([jurusan_vec], l_matrix).flatten()
                jurusan_score = round(float(np.max(sims)), 4)
            else:
                # Loker tidak minta jurusan spesifik →
                # bandingkan dengan teks loker secara umum
                if job_vec is not None:
                    jurusan_score = round(
                        float(cosine_similarity([jurusan_vec], [job_vec])[0][0]), 4
                    )
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
        pengalamans, total_pengalaman_bulan: int, lowongan, embedding_service
    ) -> float:
        minimal_bulan = lowongan.minimal_pengalaman_bulan or 0

        if minimal_bulan == 0:
            durasi_score = 1.0
        else:
            durasi_score = min(total_pengalaman_bulan / minimal_bulan, 1.0)
        durasi_score = round(durasi_score, 4)

        if not pengalamans:
            posisi_score = 0.0
        else:
            loker_vec = embedding_service.encode(
                TextPreprocessor.normalize_text(lowongan.namalowongan)
            )
            posisi_vecs = [
                embedding_service.encode(
                    TextPreprocessor.normalize_text(f"pengalaman kerja sebagai {exp.posisi}")
                )
                for exp in pengalamans
            ]
            p_matrix     = np.vstack(posisi_vecs)
            sims         = cosine_similarity(p_matrix, [loker_vec]).flatten()
            posisi_score = round(float(np.max(sims)), 4)

        return round(0.5 * posisi_score + 0.5 * durasi_score, 4)
    # ============================================================
    # FINAL SCORE — Weighted Linear Combination
    # ============================================================
    @staticmethod
    def final_score(semantic: float, skill: float, edu: float, exp: float) -> float:
        score = W_SEMANTIC * semantic + W_SKILL * skill + W_EDU * edu + W_EXP * exp
        return round(score, 4)

    # ============================================================
    # CLASSIFY — Label kecocokan
    # ============================================================


    @staticmethod
    def classify(score: float) -> str:
        """
        Threshold berdasarkan riset cosine similarity rekrutmen:
        - >= 0.80 : Sangat Cocok  → high confidence match
        - >= 0.65 : Cocok         → actionable threshold (industri: 0.65–0.72)
        - >= 0.50 : Cukup Cocok   → ada relevansi, perlu review manual
        - <  0.50 : Kurang Cocok  → tidak direkomendasikan
        """
        if score >= 0.80:
            return "Sangat Cocok"
        if score >= 0.65:
            return "Cocok"
        if score >= 0.50:
            return "Cukup Cocok"
        return "Kurang Cocok"


    @staticmethod
    def determine_color(score: float) -> str:
        """
        Green  : >= 0.65 (layak diproses)
        Yellow : >= 0.50 (perlu pertimbangan)
        Red    : <  0.50 (tidak cocok)
        """
        if score >= 0.65:
            return "green"
        if score >= 0.50:
            return "yellow"
        return "red"

    @staticmethod
    def percentage(score: float) -> int:
        return round(score * 100)
