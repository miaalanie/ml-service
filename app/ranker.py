# ranker.py
from .preprocess import TextPreprocessor
from .embedding import EmbeddingService
from .scoring import ScoringService
from .reasoning import ReasoningService


class RankerService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    def rank(self, payload) -> dict:
        lowongan = payload.lowongan
        pelamars = payload.pelamars

        if not pelamars:
            return {
                'success': True,
                'total': 0,
                'lowongan_id': lowongan.id,
                'namalowongan': lowongan.namalowongan,
                'ranked_applicants': []
            }
        
        # STEP 1 — ENCODE LOWONGAN (sekali, reuse untuk semua pelamar
        lowongan_text = TextPreprocessor.build_lowongan_text(lowongan)
        job_vec = self.embedding_service.encode(lowongan_text)

        # STEP 2 — ENCODE SEMUA PELAMAR (batch untuk efisiensi
        pelamar_texts = [
            TextPreprocessor.build_pelamar_text(p)
            for p in pelamars
        ]
        pelamar_vecs = self.embedding_service.encode_batch(pelamar_texts)

        results = []

        for i, pelamar in enumerate(pelamars):
            pelamar_vec = pelamar_vecs[i]

            # STEP 3 — S1: SEMANTIC SCORE
            semantic = ScoringService.semantic_similarity(
                pelamar_vec, job_vec
            )

            # STEP 4 — S2: SKILL SCORE
            skill = ScoringService.skill_score(
                pelamar.skills,
                job_vec,
                self.embedding_service
            )

            # STEP 5 — S3: EDUCATION SCORE
            edu = ScoringService.education_score(
                pelamar.pendidikans,
                job_vec,
                self.embedding_service
            )

            # STEP 6 — S4: EXPERIENCE SCORE
            exp = ScoringService.experience_score(
                pelamar.pengalamans,
                job_vec,
                self.embedding_service
            )

            # STEP 7 — FINAL WEIGHTED SCORE
            # w = [0.50, 0.10, 0.15, 0.25] dari ablation study
            final = ScoringService.final_score(semantic, skill, edu, exp)

            # STEP 8 — CLASSIFY & LABEL
            label      = ScoringService.classify(final)
            color      = ScoringService.determine_color(final)
            percentage = ScoringService.percentage(final)

            # STEP 9 — EXPLAINABILITY
            scores_dict = {
                'semantic': semantic,
                'skill':    skill,
                'edu':      edu,
                'exp':      exp,
            }

            tags = ReasoningService.generate_tags(
                pelamar,
                job_vec,
                self.embedding_service,
                final
            )

            reasons = ReasoningService.generate_reasons(
                pelamar,
                lowongan,
                job_vec,
                self.embedding_service,
                scores_dict
            )

            results.append({
                'pelamar_id':       pelamar.id,
                'namalengkap':      pelamar.namalengkap,

                # Score breakdown
                'match_percentage': percentage,
                'label':            label,
                'color':            color,
                'final_score':      final,
                'semantic_score':   round(semantic, 4),
                'skill_score':      round(skill, 4),
                'education_score':  round(edu, 4),
                'experience_score': round(exp, 4),

                # Explainability
                'tags':    tags,
                'reasons': reasons,
            })
        # STEP 10 — SORT DESCENDING + ASSIGN RAN
        results.sort(key=lambda x: x['final_score'], reverse=True)

        for idx, r in enumerate(results):
            r['rank'] = idx + 1

        return {
            'success':           True,
            'total':             len(results),
            'lowongan_id':       lowongan.id,
            'namalowongan':      lowongan.namalowongan,
            'ranked_applicants': results
        }