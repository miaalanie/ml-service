import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import TextPreprocessor

logger = logging.getLogger(__name__)

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
    def semantic_similarity(cv_vec: np.ndarray, job_vec: np.ndarray, log_detail: bool = False) -> float:
        if cv_vec is None or job_vec is None:
            if log_detail:
                logger.info("  [S1] Semantic Score: vector CV atau lowongan tidak tersedia -> score = 0.0")
            return 0.0
        score = cosine_similarity([cv_vec], [job_vec])[0][0]
        result = round(float(score), 4)
        if log_detail:
            logger.info("  [S1] Semantic Score: cosine_similarity(vector_CV_pelamar, vector_deskripsi_lowongan) = %.4f", result)
        return result

    # ============================================================
    # CORE — SKILL MATCH (dihitung SEKALI, dipakai scoring + reasoning)
    # ============================================================
    @staticmethod
    def compute_skill_match(
        pelamar_skills,
        lowongan_skills,
        embedding_service,
        threshold: float = SKILL_THRESHOLD,
        log_detail: bool = False,
    ) -> tuple:
        if not lowongan_skills:
            return 0.0, [], []
        if not pelamar_skills:
            return 0.0, [], [s.nama for s in lowongan_skills]

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

        if log_detail:
            logger.info("  [S2] Skill Score - threshold cosine = %.2f", threshold)
            logger.info("  [S2] Skill diminta loker (%d): %s", total_skill, [s.nama for s in lowongan_skills])
            logger.info("  [S2] Skill dimiliki pelamar (%d): %s", len(pelamar_skills), [s.namaskill for s in pelamar_skills])

        for loker_skill in lowongan_skills:
            loker_vec = ScoringService.vector_for(loker_skill)
            if loker_vec is None:
                unmatched.append(loker_skill.nama)
                if log_detail:
                    logger.info("    - '%s': embedding kosong -> tidak cocok", loker_skill.nama)
                continue

            sims     = cosine_similarity(p_matrix, [loker_vec]).flatten()
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_s   = pelamar_encoded[best_idx][0]

            if best_sim >= threshold:
                bobot = TextPreprocessor.get_skill_weight(best_s.keterangan)
                matched_score += bobot
                matched.append((loker_skill.nama, best_s.namaskill, best_s.keterangan))
                if log_detail:
                    logger.info(
                        "    - '%s' vs skill pelamar paling mirip '%s' -> cosine=%.4f (>= %.2f, COCOK), keterangan='%s' -> bobot=%.2f",
                        loker_skill.nama, best_s.namaskill, best_sim, threshold, best_s.keterangan, bobot
                    )
            else:
                unmatched.append(loker_skill.nama)
                if log_detail:
                    logger.info(
                        "    - '%s' vs skill pelamar paling mirip '%s' -> cosine=%.4f (< %.2f, TIDAK COCOK)",
                        loker_skill.nama, best_s.namaskill, best_sim, threshold
                    )

        final_result = round(matched_score / total_skill, 4)
        if log_detail:
            logger.info(
                "  [S2] Total bobot cocok = %.4f / jumlah skill loker (%d) -> skill_score = %.4f",
                matched_score, total_skill, final_result
            )
        return final_result, matched, unmatched

    # ============================================================
    # CORE — EXP FOR REASONING (dihitung SEKALI, dipakai tags + reasons)
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
        pendidikans, lowongan, job_vec: np.ndarray, embedding_service, log_detail: bool = False
    ) -> float:
        if not pendidikans:
            if log_detail:
                logger.info("  [S3] Education Score: pelamar tidak punya data pendidikan -> score = 0.2 (default)")
            return 0.2

        edu = TextPreprocessor._get_pendidikan_tertinggi(pendidikans)
        if not edu:
            if log_detail:
                logger.info("  [S3] Education Score: pendidikan tertinggi tidak ditemukan -> score = 0.2 (default)")
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
            if log_detail:
                logger.info(
                    "  [S3] Level pendidikan pelamar='%s' (level %s), syarat loker='%s' (level %s) -> level_score = min(%s/%s, 1.0) = %.4f",
                    edu.kategori, level_pelamar, lowongan.minimal_pendidikan.nama, level_minimal,
                    level_pelamar, level_minimal, min(level_score, 1.0)
                )
        else:
            level_score = 1.0
            if log_detail:
                logger.info("  [S3] Loker tidak mensyaratkan pendidikan minimum -> level_score = 1.0 (otomatis penuh)")

        level_score = round(min(level_score, 1.0), 4)

        # --- jurusan_score ---
        jurusan_pelamar = TextPreprocessor.get_jurusan_pelamar(pendidikans)

        if jurusan_pelamar is None:
            jurusan_score = 0.3
            if log_detail:
                logger.info("  [S3] Jurusan pelamar tidak diketahui -> jurusan_score = 0.3 (default)")
        else:
            jurusan_vec = ScoringService.vector_for(edu)
            if jurusan_vec is None:
                jurusan_score = 0.0
                edu_score = (0.5 * level_score) + (0.5 * jurusan_score)
                if log_detail:
                    logger.info("  [S3] Embedding jurusan pelamar kosong -> jurusan_score = 0.0")
                    logger.info(
                        "  [S3] edu_score = (0.5 x level_score %.4f) + (0.5 x jurusan_score %.4f) = %.4f",
                        level_score, jurusan_score, edu_score
                    )
                return round(edu_score, 4)

            if lowongan.jurusans:
                loker_jurusan_vecs = [ScoringService.vector_for(lj) for lj in lowongan.jurusans]
                loker_jurusan_vecs = [v for v in loker_jurusan_vecs if v is not None]
                if not loker_jurusan_vecs:
                    jurusan_score = 0.0
                    if log_detail:
                        logger.info("  [S3] Semua embedding jurusan loker kosong -> jurusan_score = 0.0")
                    return round((0.5 * level_score) + (0.5 * jurusan_score), 4)
                l_matrix = np.vstack(loker_jurusan_vecs)
                sims = cosine_similarity([jurusan_vec], l_matrix).flatten()
                jurusan_score = round(float(np.max(sims)), 4)
                if log_detail:
                    nama_jurusans = [lj.nama for lj in lowongan.jurusans]
                    logger.info(
                        "  [S3] Jurusan pelamar '%s' dibandingkan ke jurusan loker %s -> cosine tertinggi = %.4f -> jurusan_score = %.4f",
                        jurusan_pelamar, nama_jurusans, jurusan_score, jurusan_score
                    )
            else:
                if job_vec is not None:
                    jurusan_score = round(float(cosine_similarity([jurusan_vec], [job_vec])[0][0]), 4)
                    if log_detail:
                        logger.info(
                            "  [S3] Loker tidak spesifikkan jurusan, dibandingkan ke deskripsi lowongan -> cosine = %.4f",
                            jurusan_score
                        )
                else:
                    jurusan_score = 1.00
                    if log_detail:
                        logger.info("  [S3] Tidak ada data jurusan loker maupun deskripsi -> jurusan_score = 1.0 (default)")

        edu_score = (0.5 * level_score) + (0.5 * jurusan_score)
        if log_detail:
            logger.info(
                "  [S3] edu_score = (0.5 x level_score %.4f) + (0.5 x jurusan_score %.4f) = %.4f",
                level_score, jurusan_score, edu_score
            )
        return round(edu_score, 4)

    # ============================================================
    # S4 — EXPERIENCE SCORE
    # exp_score = (0.5 x posisi_score) + (0.5 x durasi_score)
    # ============================================================
    @staticmethod
    def experience_score(
        pengalamans, total_pengalaman_bulan: int, lowongan, embedding_service, log_detail: bool = False
    ) -> float:
        minimal_bulan = lowongan.minimal_pengalaman_bulan or 0

        # --- durasi_score ---
        if minimal_bulan == 0:
            durasi_score = 1.0
            if log_detail:
                logger.info("  [S4] Loker tidak mensyaratkan pengalaman minimum -> durasi_score = 1.0")
        else:
            durasi_score = min(total_pengalaman_bulan / minimal_bulan, 1.0)
            if log_detail:
                logger.info(
                    "  [S4] Total pengalaman pelamar = %s bulan, syarat minimum loker = %s bulan -> durasi_score = min(%s/%s, 1.0) = %.4f",
                    total_pengalaman_bulan, minimal_bulan, total_pengalaman_bulan, minimal_bulan, min(durasi_score, 1.0)
                )
        durasi_score = round(durasi_score, 4)

        # --- posisi_score ---
        if not pengalamans:
            posisi_score = 0.0
            if log_detail:
                logger.info("  [S4] Pelamar tidak punya riwayat pengalaman kerja -> posisi_score = 0.0")
        else:
            loker_vec = ScoringService.vector_for(lowongan, 'title_embedding')
            if loker_vec is None:
                if log_detail:
                    logger.info("  [S4] Embedding judul lowongan kosong -> posisi_score diabaikan, exp_score = 0.5 x durasi_score")
                return round(0.5 * durasi_score, 4)
            posisi_vecs = [ScoringService.vector_for(exp) for exp in pengalamans]
            posisi_vecs = [v for v in posisi_vecs if v is not None]
            if not posisi_vecs:
                if log_detail:
                    logger.info("  [S4] Semua embedding pengalaman kosong -> posisi_score diabaikan, exp_score = 0.5 x durasi_score")
                return round(0.5 * durasi_score, 4)
            p_matrix = np.vstack(posisi_vecs)
            sims = cosine_similarity(p_matrix, [loker_vec]).flatten()
            best_idx = int(np.argmax(sims))
            posisi_score = round(float(sims[best_idx]), 4)
            if log_detail:
                nama_posisi = [exp.posisi for exp in pengalamans]
                logger.info(
                    "  [S4] Posisi pengalaman pelamar %s dibandingkan ke judul lowongan '%s' -> cosine tertinggi = %.4f (dari posisi '%s')",
                    nama_posisi, lowongan.namalowongan, posisi_score, pengalamans[best_idx].posisi
                )

        exp_score = round(0.5 * posisi_score + 0.5 * durasi_score, 4)
        if log_detail:
            logger.info(
                "  [S4] exp_score = (0.5 x posisi_score %.4f) + (0.5 x durasi_score %.4f) = %.4f",
                posisi_score, durasi_score, exp_score
            )
        return exp_score

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