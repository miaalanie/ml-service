from .preprocess import TextPreprocessor
from .embedding import EmbeddingService
from .scoring import ScoringService
from .reasoning import ReasoningService


class MatcherService:
    """
    Core service untuk job matching.

    Flow:
      1. Terima payload: data pelamar + list loker aktif dari event
      2. Build representasi teks (structured text)
      3. Encode ke vektor 384 dim (sentence embedding)
      4. Hitung cosine similarity (semantic score)
      5. Hitung komponen score: skill, edu, exp
      6. Weighted scoring → final score
      7. Generate tags + reasons (explainability)
      8. Ranking berdasarkan final score
      9. Return top-N rekomendasi

    Catatan arsitektur:
      Filter loker aktif dari event dilakukan di Laravel
      sebelum payload dikirim ke ML service.
      ML service hanya menerima loker yang sudah difilter.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()

    def match(self, payload) -> dict:
        pelamar   = payload.pelamar
        lowongans = payload.lowongans

        if not lowongans:
            return {
                'success': True,
                'total': 0,
                'pelamar_id': pelamar.id,
                'recommendations': []
            }

        # ============================================================
        # STEP 1 — BUILD TEKS PELAMAR
        # Gabung skill + edu + exp menjadi 1 teks terstruktur
        # ============================================================
        pelamar_text = TextPreprocessor.build_pelamar_text(pelamar)

        # ============================================================
        # STEP 2 — ENCODE PELAMAR
        # ============================================================
        pelamar_vec = self.embedding_service.encode(pelamar_text)

        # ============================================================
        # STEP 3 — ENCODE SEMUA LOKER (batch untuk efisiensi)
        # Encode semua sekaligus lebih efisien dari satu-satu
        # ============================================================
        lowongan_texts = [
            TextPreprocessor.build_lowongan_text(lo)
            for lo in lowongans
        ]
        lowongan_vecs = self.embedding_service.encode_batch(
            lowongan_texts
        )

        results = []

        for i, lowongan in enumerate(lowongans):

            job_vec       = lowongan_vecs[i]
            lowongan_text = lowongan_texts[i]

            # ============================================================
            # STEP 4 — S1: SEMANTIC SCORE
            # Cosine similarity profil pelamar vs lowongan secara holistik
            # ============================================================
            semantic = ScoringService.semantic_similarity(
                pelamar_vec, job_vec
            )

            # ============================================================
            # STEP 5 — S2: SKILL SCORE
            # Weighted proficiency + cosine match per skill
            # ============================================================
            skill = ScoringService.skill_score(
                pelamar.skills,
                job_vec,
                self.embedding_service
            )

            # ============================================================
            # STEP 6 — S3: EDUCATION SCORE
            # 35% ordinal level + 65% semantic jurusan
            # ============================================================
            edu = ScoringService.education_score(
                pelamar.pendidikans,
                job_vec,
                self.embedding_service
            )

            # ============================================================
            # STEP 7 — S4: EXPERIENCE SCORE
            # relevance × duration_weight × recency_weight, ambil max
            # ============================================================
            exp = ScoringService.experience_score(
                pelamar.pengalamans,
                job_vec,
                self.embedding_service
            )

            # ============================================================
            # STEP 8 — FINAL SCORE
            # Weighted linear combination dari ablation study
            # w = [0.50, 0.10, 0.15, 0.25]
            # ============================================================
            final = ScoringService.final_score(
                semantic, skill, edu, exp
            )

            # ============================================================
            # STEP 9 — CLASSIFY & LABEL
            # ============================================================
            label      = ScoringService.classify(final)
            color      = ScoringService.determine_color(final)
            percentage = ScoringService.percentage(final)

            # ============================================================
            # STEP 10 — EXPLAINABILITY
            # Tags dan reasons untuk ditampilkan ke user
            # ============================================================
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
                'lowongan_id':      lowongan.id,
                'namalowongan':     lowongan.namalowongan,
                'kategori':         lowongan.kategori.nama,
                'kategorilokasi':   lowongan.kategorilokasi,
                'gaji_awal':        lowongan.gaji_awal,
                'gaji_akhir':       lowongan.gaji_akhir,

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

        # ============================================================
        # STEP 11 — RANKING
        # Urutkan berdasarkan final_score descending
        # ============================================================
        results.sort(
            key=lambda x: x['final_score'],
            reverse=True
        )

        return {
            'success':         True,
            'total':           len(results),
            'pelamar_id':      pelamar.id,
            'pelamar_nama':    pelamar.namalengkap,
            'recommendations': results
        }
