from .preprocess import TextPreprocessor
from .embedding import EmbeddingService
from .scoring import ScoringService
from .reasoning import ReasoningService


class RankerService:

    def __init__(self, embedding_service=None):
        from .embedding import EmbeddingService
        self.embedding_service = embedding_service or EmbeddingService()

    def rank(self, payload) -> dict:
        lowongan = payload.lowongan
        pelamars = payload.pelamars

        if not pelamars:
            return {
                'success':           True,
                'total':             0,
                'lowongan_id':       lowongan.id,
                'namalowongan':      lowongan.namalowongan,
                'ranked_applicants': []
            }

        # STEP 1 — BUILD & ENCODE LOKER (sekali, reuse semua pelamar)
        lowongan_text = TextPreprocessor.build_lowongan_text(lowongan)
        job_vec       = self.embedding_service.encode(lowongan_text)

        # STEP 2 — ENCODE SEMUA PELAMAR (batch)
        pelamar_texts = [
            TextPreprocessor.build_pelamar_text(p)
            for p in pelamars
        ]
        pelamar_vecs = self.embedding_service.encode_batch(pelamar_texts)

        results = []

        for i, pelamar in enumerate(pelamars):
            pelamar_vec = pelamar_vecs[i]

            # STEP 3 — BIODATA FLAGS (hard requirements langsung dari lowongan)
            biodata_flags = ReasoningService.build_biodata_flags(
                pelamar, lowongan
            )

            # STEP 4 — S1: SEMANTIC SCORE
            semantic = ScoringService.semantic_similarity(
                pelamar_vec, job_vec
            )

            # STEP 5 — Compute skill match SEKALI (dipakai scoring + reasoning)
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills,
                lowongan.skills,
                self.embedding_service
            )
            skill = skill_match[0]

            # STEP 6 — S3: EDUCATION SCORE
            edu = ScoringService.education_score(
                pelamar.pendidikans,
                lowongan,
                job_vec,
                self.embedding_service
            )

            # STEP 7 — Compute exp for reasoning SEKALI (dipakai tags + reasons)
            exp_reasoning = ScoringService.compute_exp_for_reasoning(
                pelamar.pengalamans,
                job_vec,
                self.embedding_service
            )

            # STEP 8 — S4: EXPERIENCE SCORE (tetap pakai namalowongan vector)
            exp = ScoringService.experience_score(
                pelamar.pengalamans,
                pelamar.total_pengalaman_bulan,
                lowongan,
                self.embedding_service
            )

            # STEP 9 — FINAL SCORE
            final = ScoringService.final_score(semantic, skill, edu, exp)

            # STEP 10 — CLASSIFY & LABEL
            label      = ScoringService.classify(final)
            color      = ScoringService.determine_color(final)
            percentage = ScoringService.percentage(final)

            # STEP 11 — EXPLAINABILITY (pass hasil yg sudah dihitung)
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

            reasons = ReasoningService.generate_reasons(
                pelamar, lowongan, job_vec, self.embedding_service,
                scores_dict,
                biodata_flags=biodata_flags,
                skill_match=skill_match,
                exp_reasoning=exp_reasoning,
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

        # STEP 11 — SORT DESCENDING + ASSIGN RANK
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