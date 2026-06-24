from .preprocess import TextPreprocessor
from .embedding import EmbeddingService
from .scoring import ScoringService
from .reasoning import ReasoningService


class MatcherService:

    def __init__(self, embedding_service=None):
        from .embedding import EmbeddingService
        self.embedding_service = embedding_service or EmbeddingService()

    def match(self, payload) -> dict:
        pelamar   = payload.pelamar
        lowongans = payload.lowongans

        if not lowongans:
            return {
                'success':         True,
                'total':           0,
                'pelamar_id':      pelamar.id,
                'recommendations': []
            }

        # STEP 1 — BUILD & ENCODE TEKS PELAMAR
        pelamar_text = TextPreprocessor.build_pelamar_text(pelamar)
        pelamar_vec  = self.embedding_service.encode(pelamar_text)

        # STEP 2 — ENCODE SEMUA LOKER (batch)
        lowongan_texts = [
            TextPreprocessor.build_lowongan_text(lo)
            for lo in lowongans
        ]
        lowongan_vecs = self.embedding_service.encode_batch(lowongan_texts)

        results = []

        for i, lowongan in enumerate(lowongans):
            job_vec = lowongan_vecs[i]

            # STEP 3 — BIODATA FLAGS (hard requirements langsung dari lowongan)
            biodata_flags = ReasoningService.build_biodata_flags(
                pelamar, lowongan
            )

            # STEP 4 — S1: SEMANTIC SCORE
            # Kemiripan umum teks profil pelamar vs teks loker
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

            tags = ReasoningService.generate_tags_rekomendasi(
                pelamar, lowongan, job_vec, self.embedding_service,
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
                'lowongan_id':      lowongan.id,
                'namalowongan':     lowongan.namalowongan,
                'kategori':         lowongan.kategori.nama,
                'kategorilokasi':   lowongan.kategorilokasi,
                'gaji_awal':        lowongan.gaji_awal,
                'gaji_akhir':       lowongan.gaji_akhir,
                'perusahaan_nama':  lowongan.perusahaan_nama,
                'perusahaan_logo':  lowongan.perusahaan_logo,

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

        # STEP 11 — RANKING descending
        results.sort(key=lambda x: x['final_score'], reverse=True)

        return {
            'success':         True,
            'total':           len(results),
            'pelamar_id':      pelamar.id,
            'pelamar_nama':    pelamar.namalengkap,
            'recommendations': results
        }