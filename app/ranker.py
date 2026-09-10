import logging

from .preprocess import TextPreprocessor
from .embedding import EmbeddingService
from .scoring import ScoringService
from .reasoning import ReasoningService

logger = logging.getLogger("ml-ranking")


class RankerService:

    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service or EmbeddingService()

    def rank(self, payload) -> dict:
        lowongan = payload.lowongan
        pelamars = payload.pelamars
        scoring_config = payload.scoring_config.model_dump()

        if not pelamars:
            return {
                'success':           True,
                'total':             0,
                'lowongan_id':       lowongan.id,
                'namalowongan':      lowongan.namalowongan,
                'ranked_applicants': []
            }

        job_vec = ScoringService.vector_for(lowongan)
        pelamar_vecs = [ScoringService.vector_for(p) for p in pelamars]

        results = []

        for i, pelamar in enumerate(pelamars):
            pelamar_vec = pelamar_vecs[i]

            # BIODATA FLAGS
            biodata_flags = ReasoningService.build_biodata_flags(
                pelamar, lowongan
            )

            # S1: SEMANTIC SCORE
            semantic = ScoringService.semantic_similarity(
                pelamar_vec, job_vec
            )
            logger.info(
                "S1 | pelamar_id=%s | pelamar_nama=%s | lowongan_id=%s | lowongan_nama=%s | semantic=%.4f",
                pelamar.id,
                getattr(pelamar, 'namalengkap', '-'),
                lowongan.id,
                getattr(lowongan, 'namalowongan', '-'),
                semantic,
            )

            # SKILL MATCH SEKALI (score + reasoning)
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills,
                lowongan.skills,
                self.embedding_service,
                threshold=scoring_config['skill_threshold']
            )
            skill = skill_match[0]
            logger.info(
                "S2 | pelamar_id=%s | pelamar_nama=%s | lowongan_id=%s | lowongan_nama=%s | skill=%.4f | matched=%s | unmatched=%s",
                pelamar.id,
                getattr(pelamar, 'namalengkap', '-'),
                lowongan.id,
                getattr(lowongan, 'namalowongan', '-'),
                skill,
                skill_match[1],
                skill_match[2],
            )

            # S3: EDUCATION SCORE
            edu = ScoringService.education_score(
                pelamar.pendidikans,
                lowongan,
                job_vec,
                self.embedding_service
            )
            logger.info(
                "S3 | pelamar_id=%s | pelamar_nama=%s | lowongan_id=%s | lowongan_nama=%s | edu=%.4f",
                pelamar.id,
                getattr(pelamar, 'namalengkap', '-'),
                lowongan.id,
                getattr(lowongan, 'namalowongan', '-'),
                edu,
            )

            # EXP FOR REASONING SEKALI (tags + reasons)
            exp_reasoning = ScoringService.compute_exp_for_reasoning(
                pelamar.pengalamans,
                job_vec,
                self.embedding_service
            )

            # S4: EXPERIENCE SCORE
            exp = ScoringService.experience_score(
                pelamar.pengalamans,
                pelamar.total_pengalaman_bulan,
                lowongan,
                self.embedding_service
            )
            logger.info(
                "S4 | pelamar_id=%s | pelamar_nama=%s | lowongan_id=%s | lowongan_nama=%s | exp=%.4f | best_exp=%s | best_sim=%.4f",
                pelamar.id,
                getattr(pelamar, 'namalengkap', '-'),
                lowongan.id,
                getattr(lowongan, 'namalowongan', '-'),
                exp,
                getattr(exp_reasoning[0], 'posisi', None) if exp_reasoning and exp_reasoning[0] else None,
                exp_reasoning[1] if exp_reasoning else 0.0,
            )

            # FINAL SCORE
            final = ScoringService.final_score(semantic, skill, edu, exp, scoring_config)

            # CLASSIFY & LABEL
            label      = ScoringService.classify(final)
            color      = ScoringService.determine_color(final)
            percentage = ScoringService.percentage(final)

            # EXPLAINABILITY
            scores_dict = {
                'semantic': semantic,
                'skill':    skill,
                'edu':      edu,
                'exp':      exp,
            }

            tags = ReasoningService.generate_tags(
                pelamar, lowongan, job_vec, self.embedding_service, final,
                biodata_flags=biodata_flags,
                skill_match=skill_match,
                exp_reasoning=exp_reasoning,
            )
            logger.info(
                "RANK_TAGS | pelamar_id=%s | lowongan_id=%s | tag_count=%s | preview=%s",
                pelamar.id,
                lowongan.id,
                len(tags),
                tags[:3],
            )

            reasons = ReasoningService.generate_reasons(
                pelamar, lowongan, job_vec, self.embedding_service,
                scores_dict,
                biodata_flags=biodata_flags,
                skill_match=skill_match,
                exp_reasoning=exp_reasoning,
            )
            logger.info(
                "RANK_REASONING | pelamar_id=%s | lowongan_id=%s | reason_count=%s | preview=%s",
                pelamar.id,
                lowongan.id,
                len(reasons),
                reasons[:3],
            )

            results.append({
                'pelamar_id':       pelamar.id,
                'namalengkap':      pelamar.namalengkap,

                'match_percentage': percentage,
                'label':            label,
                'color':            color,
                'final_score':      final,
                'semantic_score':   round(semantic, 4),
                'skill_score':      round(skill, 4),
                'education_score':  round(edu, 4),
                'experience_score': round(exp, 4),

                'tags':    tags,
                'reasons': reasons,
            })

        # SORT DESCENDING + ASSIGN RANK
        results.sort(key=lambda x: x['final_score'], reverse=True)

        for idx, r in enumerate(results):
            r['rank'] = idx + 1

        return {
            'success':           True,
            'total':             len(results),
            'lowongan_id':       lowongan.id,
            'namalowongan':      lowongan.namalowongan,
            'scoring_config':    scoring_config,
            'ranked_applicants': results
        }