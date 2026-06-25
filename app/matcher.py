from .preprocess import TextPreprocessor
from .embedding import EmbeddingService
from .scoring import ScoringService
from .reasoning import ReasoningService


class MatcherService:

    def __init__(self, embedding_service=None):
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

        # STEP 3 — PRE-ENCODE SEMUA TEKS UNIK SEBELUM LOOP
        # Pelamar sama untuk semua loker, encode skill/exp/jurusan sekali
        # Loker berbeda-beda, encode semua skills + jurusans seluruh loker
        texts_to_prefetch = set()

        # Pelamar skills
        for s in (pelamar.skills or []):
            texts_to_prefetch.add(
                TextPreprocessor.normalize_text(s.namaskill)
            )

        # Pelamar pengalaman
        for exp in (pelamar.pengalamans or []):
            texts_to_prefetch.add(
                TextPreprocessor.normalize_text(
                    f"pengalaman kerja sebagai {exp.posisi}"
                )
            )

        # Pelamar jurusan
        jurusan = TextPreprocessor.get_jurusan_pelamar(pelamar.pendidikans)
        if jurusan:
            texts_to_prefetch.add(
                TextPreprocessor.normalize_text(f"jurusan {jurusan}")
            )

        # Semua loker — skills, jurusans, namalowongan
        for lo in lowongans:
            texts_to_prefetch.add(
                TextPreprocessor.normalize_text(lo.namalowongan)
            )
            for ls in (lo.skills or []):
                texts_to_prefetch.add(
                    TextPreprocessor.normalize_text(ls.nama)
                )
            for lj in (lo.jurusans or []):
                texts_to_prefetch.add(
                    TextPreprocessor.normalize_text(f"jurusan {lj.nama}")
                )

        # Satu batch call → isi semua cache
        if texts_to_prefetch:
            self.embedding_service.encode_batch(list(texts_to_prefetch))

        # STEP 4 — LOOP PER LOKER (semua encode = cache hit)
        results = []

        for i, lowongan in enumerate(lowongans):
            job_vec = lowongan_vecs[i]

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

        # RANKING descending
        results.sort(key=lambda x: x['final_score'], reverse=True)

        return {
            'success':         True,
            'total':           len(results),
            'pelamar_id':      pelamar.id,
            'pelamar_nama':    pelamar.namalengkap,
            'recommendations': results
        }