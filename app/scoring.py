import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import TextPreprocessor

# BOBOT WEIGHTED SCORING
W_SEMANTIC = 0.25
W_SKILL    = 0.25
W_EDU      = 0.25
W_EXP      = 0.25

# THRESHOLD COSINE UNTUK SKILL MATCH
SKILL_THRESHOLD = 0.50


class ScoringService:

    @staticmethod
    def vector_for(item, field: str = 'embedding') -> np.ndarray | None:
        vector = getattr(item, field, None)
        if vector is None:
            return None
        result = np.asarray(vector, dtype=np.float32)
        return result if result.size == 384 else None

    # ============================================================
    # S1 — SEMANTIC SCORE
    # ============================================================
    @staticmethod
    def semantic_similarity(cv_vec: np.ndarray, job_vec: np.ndarray) -> float:
        if cv_vec is None or job_vec is None:
            return 0.0
        score = cosine_similarity([cv_vec], [job_vec])[0][0]
        return round(float(score), 4)

    # ============================================================
    # CORE — SKILL MATCH (dihitung SEKALI, dipakai scoring + reasoning)
    # Returns: (score: float, matched: list, unmatched: list)
    # matched  = [(nama_loker, nama_pelamar, keterangan), ...]
    # unmatched = [nama_skill_loker yang tidak ter-cover, ...]
    # ============================================================
    @staticmethod
    def compute_skill_match(
        pelamar_skills,
        lowongan_skills,
        embedding_service,
        threshold: float = SKILL_THRESHOLD,
    ) -> tuple:
        if not lowongan_skills:
            return 0.0, [], []
        if not pelamar_skills:
            return 0.0, [], [s.nama for s in lowongan_skills]

        # Encode semua skill pelamar sekali
        pelamar_encoded = [
            (s, ScoringService.vector_for(s))
            for s in pelamar_skills
        ]
        pelamar_encoded = [(s, vector) for s, vector in pelamar_encoded if vector is not None]
        if not pelamar_encoded:
            return 0.0, [], [s.nama for s in lowongan_skills]
        p_matrix = np.vstack([v for _, v in pelamar_encoded])

        total_skill   = len(lowongan_skills)
        matched_score = 0.0
        matched       = []
        unmatched     = []

        for loker_skill in lowongan_skills:
            loker_vec = ScoringService.vector_for(loker_skill)
            if loker_vec is None:
                unmatched.append(loker_skill.nama)
                continue
            # Batch cosine: semua skill pelamar vs satu skill loker sekaligus
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

    # ============================================================
    # CORE — EXP FOR REASONING (dihitung SEKALI, dipakai tags + reasons)
    # Returns: (best_exp, best_sim)
    # ============================================================
    @staticmethod
    def compute_exp_for_reasoning(
        pengalamans,
        job_vec: np.ndarray,
        embedding_service,
    ) -> tuple:
        if not pengalamans:
            return None, 0.0

        pengalaman_vectors = [
            (exp, ScoringService.vector_for(exp))
            for exp in pengalamans
        ]
        pengalaman_vectors = [
            (exp, vector) for exp, vector in pengalaman_vectors if vector is not None
        ]
        if not pengalaman_vectors or job_vec is None:
            return None, 0.0
        p_matrix = np.vstack([vector for _, vector in pengalaman_vectors])
        sims     = cosine_similarity(p_matrix, [job_vec]).flatten()
        best_idx = int(np.argmax(sims))
        return pengalaman_vectors[best_idx][0], round(float(sims[best_idx]), 4)


    # ============================================================
    # S3 — EDUCATION SCORE
    # edu_score = (0.5 x level_score) + (0.5 x jurusan_score)
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
            # Jika lowongan bebas pendidikan, semua pelamar otomatis dapet nilai sempurna (1.0)
            level_score = 1.0

        level_score = round(min(level_score, 1.0), 4)

        # --- jurusan_score ---
        jurusan_pelamar = TextPreprocessor.get_jurusan_pelamar(pendidikans)

        if jurusan_pelamar is None:
            jurusan_score = 0.3
        else:
            jurusan_vec  = ScoringService.vector_for(edu)
            if jurusan_vec is None:
                jurusan_score = 0.0
                edu_score = (0.5 * level_score) + (0.5 * jurusan_score)
                return round(edu_score, 4)

            if lowongan.jurusans:
                # Batch cosine: jurusan pelamar vs semua jurusan loker sekaligus
                loker_jurusan_vecs = [
                        ScoringService.vector_for(lj)
                    for lj in lowongan.jurusans
                ]
                loker_jurusan_vecs = [vector for vector in loker_jurusan_vecs if vector is not None]
                if not loker_jurusan_vecs:
                    jurusan_score = 0.0
                    return round((0.5 * level_score) + (0.5 * jurusan_score), 4)
                l_matrix      = np.vstack(loker_jurusan_vecs)
                sims          = cosine_similarity([jurusan_vec], l_matrix).flatten()
                jurusan_score = round(float(np.max(sims)), 4)
            else:
                if job_vec is not None:
                    jurusan_score = round(
                        float(cosine_similarity([jurusan_vec], [job_vec])[0][0]), 4
                    )
                else:
                    jurusan_score = 1.00

        edu_score = (0.5 * level_score) + (0.5 * jurusan_score)
        return round(edu_score, 4)

    # ============================================================
    # S4 — EXPERIENCE SCORE
    # exp_score = (0.5 x posisi_score) + (0.5 x durasi_score)
    # ============================================================
    @staticmethod
    def experience_score(
        pengalamans, total_pengalaman_bulan: int, lowongan, embedding_service
    ) -> float:
        minimal_bulan = lowongan.minimal_pengalaman_bulan or 0

        # --- durasi_score ---
        if minimal_bulan == 0:
            durasi_score = 1.0
        else:
            durasi_score = min(total_pengalaman_bulan / minimal_bulan, 1.0)
        durasi_score = round(durasi_score, 4)

        # --- posisi_score ---
        if not pengalamans:
            posisi_score = 0.0
        else:
            loker_vec = ScoringService.vector_for(lowongan, 'title_embedding')
            if loker_vec is None:
                return round(0.5 * durasi_score, 4)
            posisi_vecs = [ScoringService.vector_for(exp) for exp in pengalamans]
            posisi_vecs = [vector for vector in posisi_vecs if vector is not None]
            if not posisi_vecs:
                return round(0.5 * durasi_score, 4)
            p_matrix     = np.vstack(posisi_vecs)
            sims         = cosine_similarity(p_matrix, [loker_vec]).flatten()
            posisi_score = round(float(np.max(sims)), 4)

        return round(0.5 * posisi_score + 0.5 * durasi_score, 4)

    # ============================================================
    # FINAL SCORE — Weighted Linear Combination
    # ============================================================
    @staticmethod
    def final_score(semantic: float, skill: float, edu: float, exp: float, config=None) -> float:
        config = config or {}
        score = (
            config.get('semantic_weight', W_SEMANTIC) * semantic
            + config.get('skill_weight', W_SKILL) * skill
            + config.get('education_weight', W_EDU) * edu
            + config.get('experience_weight', W_EXP) * exp
        )
        return round(score, 4)

    # ============================================================
    # CLASSIFY — Label kecocokan
    # ============================================================
    @staticmethod
    def classify(score: float) -> str:
        if score >= 0.80:
            return "Sangat Cocok"
        if score >= 0.65:
            return "Cocok"
        if score >= 0.50:
            return "Cukup Cocok"
        return "Kurang Cocok"

    @staticmethod
    def determine_color(score: float) -> str:
        if score >= 0.65:
            return "green"
        if score >= 0.50:
            return "yellow"
        return "red"

    @staticmethod
    def percentage(score: float) -> int:
        return round(score * 100)