from .preprocess import TextPreprocessor
from .embedding import EmbeddingService
from .scoring import ScoringService
from .reasoning import ReasoningService


class RankerService:

    def __init__(self, embedding_service=None):
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

        # STEP 3 — PRE-ENCODE SEMUA TEKS UNIK SEBELUM LOOP
        # Kumpulkan semua teks dari loker + seluruh pelamar,
        # encode sekaligus dalam satu batch → loop jadi pure cache hit
        texts_to_prefetch = set()

        # Loker skills
        for ls in (lowongan.skills or []):
            texts_to_prefetch.add(
                TextPreprocessor.normalize_text(ls.nama)
            )

        # Loker jurusans
        for lj in (lowongan.jurusans or []):
            texts_to_prefetch.add(
                TextPreprocessor.normalize_text(f"jurusan {lj.nama}")
            )

        # Namalowongan (dipakai experience_score)
        texts_to_prefetch.add(
            TextPreprocessor.normalize_text(lowongan.namalowongan)
        )

        # Semua pelamar — skills, pengalaman, jurusan
        for p in pelamars:
            for s in (p.skills or []):
                texts_to_prefetch.add(
                    TextPreprocessor.normalize_text(s.namaskill)
                )
            for exp in (p.pengalamans or []):
                texts_to_prefetch.add(
                    TextPreprocessor.normalize_text(
                        f"pengalaman kerja sebagai {exp.posisi}"
                    )
                )
            jurusan = TextPreprocessor.get_jurusan_pelamar(p.pendidikans)
            if jurusan:
                texts_to_prefetch.add(
                    TextPreprocessor.normalize_text(f"jurusan {jurusan}")
                )

        # Satu batch call → isi semua cache
        if texts_to_prefetch:
            self.embedding_service.encode_batch(list(texts_to_prefetch))

        # STEP 4 — LOOP PER PELAMAR (semua encode = cache hit)
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

            # SKILL MATCH SEKALI (score + reasoning)
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills,
                lowongan.skills,
                self.embedding_service
            )
            skill = skill_match[0]

            # S3: EDUCATION SCORE
            edu = ScoringService.education_score(
                pelamar.pendidikans,
                lowongan,
                job_vec,
                self.embedding_service
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

            # FINAL SCORE
            final = ScoringService.final_score(semantic, skill, edu, exp)

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
            'ranked_applicants': results
        }