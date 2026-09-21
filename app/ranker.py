import logging
from .preprocess import TextPreprocessor
from .embedding import EmbeddingService
from .scoring import ScoringService
from .reasoning import ReasoningService

logger = logging.getLogger(__name__)


class RankerService:
    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service or EmbeddingService()

    def rank(self, payload) -> dict:
        lowongan = payload.lowongan
        pelamars = payload.pelamars
        scoring_config = payload.scoring_config.model_dump()

        if not pelamars:
            return {
                "success": True,
                "total": 0,
                "lowongan_id": lowongan.id,
                "namalowongan": lowongan.namalowongan,
                "ranked_applicants": [],
            }

        job_vec = ScoringService.vector_for(lowongan)
        pelamar_vecs = [ScoringService.vector_for(p) for p in pelamars]

        results = []
        MAX_LOG_DETAIL = 20  # cuma 3 pelamar pertama yang di-log detail

        logger.info(" ")
        logger.info(
            "========== SCORING DETAIL | Lowongan #%s (%s) vs %d Pelamar ==========",
            lowongan.id,
            lowongan.namalowongan,
            len(pelamars),
        )
        logger.info(
            "Catatan: detail perhitungan S1-S4 hanya ditampilkan untuk %d pelamar pertama sebagai contoh.",
            MAX_LOG_DETAIL,
        )

        for i, pelamar in enumerate(pelamars):
            pelamar_vec = pelamar_vecs[i]
            log_detail = i < MAX_LOG_DETAIL

            if log_detail:
                logger.info(" ")
                logger.info(
                    "===================== Pelamar #%s: %s =====================",
                    pelamar.id,
                    pelamar.namalengkap,
                )

            # BIODATA FLAGS
            biodata_flags = ReasoningService.build_biodata_flags(
                pelamar, lowongan, log_detail=log_detail
            )

            # S1: SEMANTIC SCORE
            semantic = ScoringService.semantic_similarity(
                pelamar_vec, job_vec, log_detail=log_detail
            )

            # SKILL MATCH SEKALI (score + reasoning)
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills,
                lowongan.skills,
                self.embedding_service,
                threshold=scoring_config["skill_threshold"],
                log_detail=log_detail,
            )
            skill = skill_match[0]

            # S3: EDUCATION SCORE
            edu = ScoringService.education_score(
                pelamar.pendidikans,
                lowongan,
                job_vec,
                self.embedding_service,
                log_detail=log_detail,
            )

            # EXP FOR REASONING SEKALI (tags + reasons)
            exp_reasoning = ScoringService.compute_exp_for_reasoning(
                pelamar.pengalamans, job_vec, self.embedding_service
            )

            # S4: EXPERIENCE SCORE
            exp = ScoringService.experience_score(
                pelamar.pengalamans,
                pelamar.total_pengalaman_bulan,
                lowongan,
                self.embedding_service,
                log_detail=log_detail,
            )

            # FINAL SCORE
            final = ScoringService.final_score(
                semantic, skill, edu, exp, scoring_config
            )

            if log_detail:
                logger.info(
                    "  [FINAL] final_score = (%.2f x semantic %.4f) + (%.2f x skill %.4f) + (%.2f x edu %.4f) + (%.2f x exp %.4f) = %.4f",
                    scoring_config.get("semantic_weight", 0.25),
                    semantic,
                    scoring_config.get("skill_weight", 0.25),
                    skill,
                    scoring_config.get("education_weight", 0.25),
                    edu,
                    scoring_config.get("experience_weight", 0.25),
                    exp,
                    final,
                )

            # CLASSIFY & LABEL
            label = ScoringService.classify(final)
            color = ScoringService.determine_color(final)
            percentage = ScoringService.percentage(final)

            # EXPLAINABILITY
            scores_dict = {
                "semantic": semantic,
                "skill": skill,
                "edu": edu,
                "exp": exp,
            }

            tags = ReasoningService.generate_tags(
                pelamar,
                lowongan,
                job_vec,
                self.embedding_service,
                final,
                biodata_flags=biodata_flags,
                skill_match=skill_match,
                exp_reasoning=exp_reasoning,
                log_detail=log_detail,
            )

            reasons = ReasoningService.generate_reasons(
                pelamar,
                lowongan,
                job_vec,
                self.embedding_service,
                scores_dict,
                biodata_flags=biodata_flags,
                skill_match=skill_match,
                exp_reasoning=exp_reasoning,
                log_detail=log_detail,
            )

            results.append(
                {
                    "pelamar_id": pelamar.id,
                    "namalengkap": pelamar.namalengkap,
                    "match_percentage": percentage,
                    "label": label,
                    "color": color,
                    "final_score": final,
                    "semantic_score": round(semantic, 4),
                    "skill_score": round(skill, 4),
                    "education_score": round(edu, 4),
                    "experience_score": round(exp, 4),
                    "tags": tags,
                    "reasons": reasons,
                }
            )

        # SORT DESCENDING + ASSIGN RANK
        results.sort(key=lambda x: x["final_score"], reverse=True)

        for idx, r in enumerate(results):
            r["rank"] = idx + 1

        return {
            "success": True,
            "total": len(results),
            "lowongan_id": lowongan.id,
            "namalowongan": lowongan.namalowongan,
            "scoring_config": scoring_config,
            "ranked_applicants": results,
        }
