from app.preprocess import TextPreprocessor
from app.embedding import EmbeddingService
from app.scoring import ScoringService
from app.reasoning import ReasoningService


class MatcherService:

    def __init__(self):

        self.embedding_service = (
            EmbeddingService()
        )

    def match(self, payload):

        pelamar = payload.pelamar
        lowongans = payload.lowongans

        # =========================
        # BUILD PELAMAR TEXT
        # =========================

        pelamar_text = (
            TextPreprocessor.build_pelamar_text(
                pelamar
            )
        )

        # =========================
        # CREATE CV VECTOR
        # =========================

        pelamar_vector = (
            self.embedding_service.encode(
                pelamar_text
            )
        )

        results = []

        # =========================
        # LOOP LOWONGAN
        # =========================

        for lowongan in lowongans:

            # =========================
            # BUILD LOWONGAN TEXT
            # =========================

            lowongan_text = (
                TextPreprocessor.build_lowongan_text(
                    lowongan
                )
            )

            # =========================
            # CREATE JOB VECTOR
            # =========================

            lowongan_vector = (
                self.embedding_service.encode(
                    lowongan_text
                )
            )

            # =========================
            # SEMANTIC SIMILARITY
            # =========================

            semantic_score = (
                ScoringService.semantic_similarity(
                    pelamar_vector,
                    lowongan_vector
                )
            )

            # =========================
            # SKILL MATCH
            # =========================

            skill_score = (
                ScoringService.skill_match(
                    [
                        skill.namaskill
                        for skill in pelamar.skills
                    ],
                    lowongan_text
                )
            )

            # =========================
            # EDUCATION MATCH
            # =========================

            edu_score = (
                ScoringService.education_match(
                    pelamar.pendidikans,
                    lowongan_text
                )
            )

            # =========================
            # EXPERIENCE MATCH
            # =========================

            exp_score = (
                ScoringService.experience_match(
                    pelamar.pengalamans,
                    lowongan_text
                )
            )

            # =========================
            # FINAL SCORE
            # =========================

            final_score = (
                ScoringService.final_score(
                    semantic_score,
                    skill_score,
                    edu_score,
                    exp_score
                )
            )

            # =========================
            # LABEL
            # =========================

            label = (
                ScoringService.classify(
                    final_score
                )
            )

            # =========================
            # TAGS
            # =========================

            tags = (
                ReasoningService.generate_tags(
                    pelamar,
                    lowongan_text,
                    final_score
                )
            )

            # =========================
            # REASONS
            # =========================

            reasons = (
                ReasoningService.generate_reasons(
                    pelamar,
                    lowongan_text
                )
            )

            # =========================
            # COLOR
            # =========================

            color = (
                ReasoningService.determine_color(
                    final_score
                )
            )

            # =========================
            # PERCENTAGE
            # =========================

            percentage = (
                ReasoningService.percentage(
                    final_score
                )
            )

            # =========================
            # COMPANY
            # =========================

            company_name = None

            try:
                company_name = (
                    lowongan
                    .register
                    .perusahaan
                    .nama
                )
            except:
                company_name = None

            # =========================
            # APPEND RESULT
            # =========================

            results.append({

                'lowongan_id':
                    lowongan.id,

                'namalowongan':
                    lowongan.namalowongan,

                'company':
                    company_name,

                'kategori':
                    lowongan.kategori.nama,

                'match_percentage':
                    percentage,

                'label':
                    label,

                'color':
                    color,

                'semantic_score':
                    round(semantic_score, 2),

                'skill_score':
                    round(skill_score, 2),

                'education_score':
                    round(edu_score, 2),

                'experience_score':
                    round(exp_score, 2),

                'final_score':
                    final_score,

                'tags':
                    tags,

                'reasons':
                    reasons
            })

        # =========================
        # SORTING
        # =========================

        results.sort(
            key=lambda x: x['final_score'],
            reverse=True
        )

        # =========================
        # RETURN
        # =========================

        return {
            'success': True,
            'total': len(results),
            'recommendations': results
        }