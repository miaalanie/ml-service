import logging
from datetime import date, datetime

from .preprocess import TextPreprocessor
from .scoring import ScoringService
from .biodata_validator import BiodataValidator

logger = logging.getLogger(__name__)
_validator = BiodataValidator()

CURRENT_YEAR = datetime.now().year

EDU_LABEL_MAP = {
    9: 'S3', 8: 'S2', 7: 'D4/S1', 6: 'D3',
    5: 'D2', 4: 'D1', 3: 'SMA/SMK', 2: 'SMP', 1: 'SD'
}


def _bulan_to_label(bulan: int) -> str:
    if bulan == 0:
        return '< 1 bulan'
    if bulan < 12:
        return f'{bulan} bulan'
    tahun = bulan // 12
    sisa  = bulan % 12
    if sisa == 0:
        return f'±{tahun} tahun'
    return f'±{tahun} tahun {sisa} bulan'


def _get_edu_tertinggi(pelamar):
    if not pelamar.pendidikans:
        return None
    return max(
        pelamar.pendidikans,
        key=lambda p: TextPreprocessor.get_pendidikan_level(p.kategori)
    )


def _get_skills_text(pelamar) -> list:
    return [s.namaskill for s in (pelamar.skills or [])]


def _get_min_edu_req(lowongan) -> int:
    if lowongan.minimal_pendidikan and lowongan.minimal_pendidikan.kode:
        return TextPreprocessor.get_pendidikan_level_from_kode(
            lowongan.minimal_pendidikan.kode
        )
    return 0


class ReasoningService:

    # ============================================================
    # TAGS — untuk /rank-applicants (company view)
    # ============================================================
    @staticmethod
    def generate_tags(
        pelamar,
        lowongan,
        job_vec,
        embedding_service,
        final_score: float,
        biodata_flags: dict = None,
        skill_match: tuple = None,
        exp_reasoning: tuple = None,
        log_detail: bool = False,
    ) -> list:
        tags = []

        if log_detail:
            logger.info("  [TAGS] Menyusun tag untuk company view (generate_tags):")

        # TAG 1 — BIODATA
        if biodata_flags:
            gm = biodata_flags.get('gender_match')
            um = biodata_flags.get('usia_match')

            if gm is True:
                text = biodata_flags.get('gender_note', 'Gender sesuai')
                tags.append({'type': 'success', 'text': text})
                if log_detail:
                    logger.info("    - TAG success: '%s' (gender_match=True)", text)
            elif gm is False:
                text = biodata_flags.get('gender_note', 'Gender tidak sesuai')
                tags.append({'type': 'danger', 'text': text})
                if log_detail:
                    logger.info("    - TAG danger: '%s' (gender_match=False)", text)

            if um is True:
                usia = biodata_flags.get('usia')
                text = f'Usia {usia} tahun — sesuai syarat'
                tags.append({'type': 'success', 'text': text})
                if log_detail:
                    logger.info("    - TAG success: '%s' (usia_match=True)", text)
            elif um is False:
                text = biodata_flags.get('usia_note', 'Usia tidak memenuhi syarat')
                tags.append({'type': 'danger', 'text': text})
                if log_detail:
                    logger.info("    - TAG danger: '%s' (usia_match=False)", text)

        # TAG 2 — SKILL
        if skill_match is None:
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills, lowongan.skills, embedding_service
            )
        _, matched_skills, _ = skill_match

        if not lowongan.skills:
            if log_detail:
                logger.info("    - TAG skill: dilewati (loker tidak minta skill apapun)")
        elif not pelamar.skills:
            tags.append({'type': 'warning', 'text': 'Belum ada skill terdaftar di profil'})
            if log_detail:
                logger.info("    - TAG warning: 'Belum ada skill terdaftar di profil' (pelamar.skills kosong)")
        elif matched_skills:
            for loker_nama, _, ket in matched_skills[:3]:
                text = f'{loker_nama} ({ket})'
                tags.append({'type': 'success', 'text': text})
                if log_detail:
                    logger.info("    - TAG success: '%s' (dari matched_skills, max 3 ditampilkan)", text)
        else:
            tags.append({'type': 'danger', 'text': 'Skill belum relevan dengan kebutuhan loker'})
            if log_detail:
                logger.info("    - TAG danger: 'Skill belum relevan dengan kebutuhan loker' (matched_skills kosong)")

        # TAG 3 — BAHASA
        if biodata_flags:
            lang_missing = biodata_flags.get('lang_missing', [])
            lang_found   = biodata_flags.get('lang_found', [])
            if lang_missing:
                text = f'Tidak ada bukti kemampuan: {", ".join(lang_missing)}'
                tags.append({'type': 'danger', 'text': text})
                if log_detail:
                    logger.info("    - TAG danger: '%s' (lang_missing=%s)", text, lang_missing)
            elif lang_found:
                text = f'Bahasa {", ".join(lang_found)} tersedia'
                tags.append({'type': 'success', 'text': text})
                if log_detail:
                    logger.info("    - TAG success: '%s' (lang_found=%s)", text, lang_found)

        # TAG 4 — PENDIDIKAN
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        min_edu_req   = _get_min_edu_req(lowongan)

        if not pelamar.pendidikans:
            tags.append({'type': 'warning', 'text': 'Data pendidikan tidak tersedia'})
            if log_detail:
                logger.info("    - TAG warning: 'Data pendidikan tidak tersedia'")
        else:
            edu_level = TextPreprocessor.get_pendidikan_level(edu_tertinggi.kategori)
            edu_label = edu_tertinggi.kategori

            if min_edu_req == 0:
                text = f'Pendidikan: {edu_label}'
                tags.append({'type': 'info', 'text': text})
                if log_detail:
                    logger.info("    - TAG info: '%s' (loker tidak syaratkan pendidikan)", text)
            elif edu_level >= min_edu_req:
                text = f'Pendidikan {edu_label} — memenuhi syarat'
                tags.append({'type': 'success', 'text': text})
                if log_detail:
                    logger.info("    - TAG success: '%s' (level %s >= syarat %s)", text, edu_level, min_edu_req)
            else:
                text = f'Pendidikan {edu_label} — di bawah syarat minimum'
                tags.append({'type': 'danger', 'text': text})
                if log_detail:
                    logger.info("    - TAG danger: '%s' (level %s < syarat %s)", text, edu_level, min_edu_req)

        # TAG 5 — PENGALAMAN
        if exp_reasoning is None:
            exp_reasoning = ScoringService.compute_exp_for_reasoning(
                pelamar.pengalamans, job_vec, embedding_service
            )
        best_exp, best_sim = exp_reasoning

        total_bulan   = pelamar.total_pengalaman_bulan
        min_exp_bulan = lowongan.minimal_pengalaman_bulan or 0

        if not pelamar.pengalamans:
            if min_exp_bulan > 0:
                text = f'Tidak ada pengalaman (loker min. {_bulan_to_label(min_exp_bulan)})'
                tags.append({'type': 'danger', 'text': text})
                if log_detail:
                    logger.info("    - TAG danger: '%s'", text)
            else:
                tags.append({'type': 'warning', 'text': 'Belum ada pengalaman kerja'})
                if log_detail:
                    logger.info("    - TAG warning: 'Belum ada pengalaman kerja' (loker tidak syaratkan exp)")
        else:
            if best_sim >= 0.80:
                rel_label, rel_type = "sangat relevan", "success"
            elif best_sim >= 0.65:
                rel_label, rel_type = "relevan", "success"
            elif best_sim >= 0.50:
                rel_label, rel_type = "cukup relevan", "warning"
            else:
                rel_label, rel_type = "kurang relevan", "warning"

            dur_label = _bulan_to_label(total_bulan)

            if min_exp_bulan > 0 and total_bulan < min_exp_bulan:
                text = f'Pengalaman {dur_label} — kurang dari syarat {_bulan_to_label(min_exp_bulan)}'
                tags.append({'type': 'warning', 'text': text})
                if log_detail:
                    logger.info("    - TAG warning: '%s' (%s bulan < syarat %s bulan)", text, total_bulan, min_exp_bulan)
            else:
                text = f'Pengalaman {dur_label} ({rel_label})'
                tags.append({'type': rel_type, 'text': text})
                if log_detail:
                    logger.info("    - TAG %s: '%s' (best_sim=%.4f -> %s)", rel_type, text, best_sim, rel_label)

        return tags

    # ============================================================
    # TAGS — untuk /match (job seeker view)
    # ============================================================
    @staticmethod
    def generate_tags_rekomendasi(
        pelamar,
        lowongan,
        job_vec,
        embedding_service,
        biodata_flags: dict = None,
        skill_match: tuple = None,
        exp_reasoning: tuple = None,
        log_detail: bool = False,
    ) -> list:
        tags = []

        if log_detail:
            logger.info("  [TAGS] Menyusun tag untuk job seeker view (generate_tags_rekomendasi):")

        # TAG 1 — BIODATA
        if biodata_flags:
            gm = biodata_flags.get('gender_match')
            um = biodata_flags.get('usia_match')

            if gm is True:
                tags.append({'type': 'success', 'text': 'Gender sesuai'})
                if log_detail:
                    logger.info("    - TAG success: 'Gender sesuai' (gender_match=True)")
            elif gm is False:
                tags.append({'type': 'danger', 'text': 'Gender tidak sesuai'})
                if log_detail:
                    logger.info("    - TAG danger: 'Gender tidak sesuai' (gender_match=False)")

            if um is True:
                usia = biodata_flags.get('usia')
                text = f'Usia {usia} thn sesuai'
                tags.append({'type': 'success', 'text': text})
                if log_detail:
                    logger.info("    - TAG success: '%s' (usia_match=True)", text)
            elif um is False:
                tags.append({'type': 'danger', 'text': 'Usia tidak memenuhi syarat'})
                if log_detail:
                    logger.info("    - TAG danger: 'Usia tidak memenuhi syarat' (usia_match=False)")

        # TAG 2 — SKILL
        if skill_match is None:
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills, lowongan.skills, embedding_service
            )
        _, matched_skills, _ = skill_match

        if not lowongan.skills:
            if log_detail:
                logger.info("    - TAG skill: dilewati (loker tidak minta skill apapun)")
        elif not pelamar.skills:
            tags.append({'type': 'warning', 'text': 'Belum ada skill'})
            if log_detail:
                logger.info("    - TAG warning: 'Belum ada skill' (pelamar.skills kosong)")
        elif matched_skills:
            tags.append({'type': 'success', 'text': 'Skill relevan'})
            if log_detail:
                logger.info("    - TAG success: 'Skill relevan' (matched_skills ada %d item)", len(matched_skills))
        else:
            tags.append({'type': 'warning', 'text': 'Skill kurang sesuai'})
            if log_detail:
                logger.info("    - TAG warning: 'Skill kurang sesuai' (matched_skills kosong)")

        # TAG 3 — BAHASA
        if biodata_flags:
            lang_missing = biodata_flags.get('lang_missing', [])
            lang_found   = biodata_flags.get('lang_found', [])
            if lang_missing:
                text = f'Perlu {", ".join(lang_missing)}'
                tags.append({'type': 'danger', 'text': text})
                if log_detail:
                    logger.info("    - TAG danger: '%s' (lang_missing=%s)", text, lang_missing)
            elif lang_found:
                text = f'Bahasa {", ".join(lang_found)} ✓'
                tags.append({'type': 'success', 'text': text})
                if log_detail:
                    logger.info("    - TAG success: '%s' (lang_found=%s)", text, lang_found)

        # TAG 4 — PENDIDIKAN
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        min_edu_req   = _get_min_edu_req(lowongan)

        if not pelamar.pendidikans:
            tags.append({'type': 'warning', 'text': 'Data pendidikan kosong'})
            if log_detail:
                logger.info("    - TAG warning: 'Data pendidikan kosong'")
        else:
            edu_level = TextPreprocessor.get_pendidikan_level(edu_tertinggi.kategori)
            if min_edu_req == 0:
                tags.append({'type': 'info', 'text': 'Pendidikan tidak disyaratkan'})
                if log_detail:
                    logger.info("    - TAG info: 'Pendidikan tidak disyaratkan'")
            elif edu_level >= min_edu_req:
                text = f'Pendidikan {edu_tertinggi.kategori} ✓'
                tags.append({'type': 'success', 'text': text})
                if log_detail:
                    logger.info("    - TAG success: '%s' (level %s >= syarat %s)", text, edu_level, min_edu_req)
            else:
                req_label = EDU_LABEL_MAP.get(min_edu_req, str(min_edu_req))
                text = f'Pendidikan Anda di bawah syarat {req_label}'
                tags.append({'type': 'danger', 'text': text})
                if log_detail:
                    logger.info("    - TAG danger: '%s' (level %s < syarat %s)", text, edu_level, min_edu_req)

        # TAG 5 — PENGALAMAN
        if exp_reasoning is None:
            exp_reasoning = ScoringService.compute_exp_for_reasoning(
                pelamar.pengalamans, job_vec, embedding_service
            )
        best_exp, best_sim = exp_reasoning

        total_bulan   = pelamar.total_pengalaman_bulan
        min_exp_bulan = lowongan.minimal_pengalaman_bulan or 0

        if not pelamar.pengalamans:
            if min_exp_bulan == 0:
                tags.append({'type': 'success', 'text': 'Fresh graduate welcome'})
                if log_detail:
                    logger.info("    - TAG success: 'Fresh graduate welcome' (tidak ada exp, loker tidak syaratkan exp)")
            else:
                text = f'Perlu pengalaman min. {_bulan_to_label(min_exp_bulan)}'
                tags.append({'type': 'danger', 'text': text})
                if log_detail:
                    logger.info("    - TAG danger: '%s'", text)
        else:
            if min_exp_bulan == 0:
                tags.append({'type': 'success', 'text': 'Fresh graduate welcome'})
                if log_detail:
                    logger.info("    - TAG success: 'Fresh graduate welcome' (loker tidak syaratkan exp minimum)")
            elif total_bulan < min_exp_bulan:
                text = f'Pengalaman {_bulan_to_label(total_bulan)} (min. {_bulan_to_label(min_exp_bulan)})'
                tags.append({'type': 'warning', 'text': text})
                if log_detail:
                    logger.info("    - TAG warning: '%s' (%s bulan < syarat %s bulan)", text, total_bulan, min_exp_bulan)
            elif best_sim >= 0.45:
                tags.append({'type': 'success', 'text': 'Pengalaman relevan'})
                if log_detail:
                    logger.info("    - TAG success: 'Pengalaman relevan' (best_sim=%.4f >= 0.45)", best_sim)
            elif best_sim >= 0.30:
                tags.append({'type': 'warning', 'text': 'Pengalaman cukup relevan'})
                if log_detail:
                    logger.info("    - TAG warning: 'Pengalaman cukup relevan' (best_sim=%.4f)", best_sim)
            else:
                tags.append({'type': 'warning', 'text': 'Pengalaman kurang relevan'})
                if log_detail:
                    logger.info("    - TAG warning: 'Pengalaman kurang relevan' (best_sim=%.4f < 0.30)", best_sim)

        return tags

    # ============================================================
    # REASONS — dipakai kedua endpoint
    # ============================================================
    @staticmethod
    def generate_reasons(
        pelamar,
        lowongan,
        job_vec,
        embedding_service,
        scores: dict,
        biodata_flags: dict = None,
        skill_match: tuple = None,
        exp_reasoning: tuple = None,
        log_detail: bool = False,
    ) -> list:
        reasons = []

        if log_detail:
            logger.info("  [REASONS] Menyusun kalimat penjelasan (generate_reasons):")

        # 1. SEMANTIC
        sem = scores.get('semantic', 0)
        if sem >= 0.80:
            r = (f"Profil pelamar secara keseluruhan sangat sesuai dengan "
                 f"deskripsi lowongan {lowongan.namalowongan} "
                 f"(tingkat kesesuaian {round(sem * 100)}%).")
        elif sem >= 0.65:
            r = (f"Profil pelamar sesuai dengan deskripsi lowongan "
                 f"{lowongan.namalowongan} "
                 f"(tingkat kesesuaian {round(sem * 100)}%).")
        elif sem >= 0.50:
            r = (f"Profil pelamar cukup sesuai dengan deskripsi lowongan "
                 f"{lowongan.namalowongan} "
                 f"(tingkat kesesuaian {round(sem * 100)}%).")
        else:
            r = (f"Kesesuaian profil pelamar dengan deskripsi lowongan "
                 f"{lowongan.namalowongan} masih rendah "
                 f"(tingkat kesesuaian {round(sem * 100)}%). "
                 f"Profil pelamar mungkin perlu dilengkapi lebih lanjut.")
        reasons.append(r)
        if log_detail:
            logger.info("    1. [semantic=%.4f] %s", sem, r)

        # 2. BIODATA
        if biodata_flags:
            gm = biodata_flags.get('gender_match')
            um = biodata_flags.get('usia_match')

            if gm is False:
                r = biodata_flags['gender_note'] + '.'
                reasons.append(r)
                if log_detail:
                    logger.info("    2a. [gender_match=False] %s", r)
            if um is False:
                r = biodata_flags['usia_note'] + '.'
                reasons.append(r)
                if log_detail:
                    logger.info("    2b. [usia_match=False] %s", r)
            if gm is True and um is True:
                usia       = biodata_flags.get('usia')
                gender_val = getattr(pelamar, 'jeniskelamin', '') or ''
                r = (f"Pelamar memenuhi syarat biodata loker: "
                     f"jenis kelamin {gender_val} dan usia {usia} tahun sesuai ketentuan.")
                reasons.append(r)
                if log_detail:
                    logger.info("    2c. [gender_match=True, usia_match=True] %s", r)
            elif gm is None and um is True:
                usia = biodata_flags.get('usia')
                r = f"Usia pelamar ({usia} tahun) sesuai dengan ketentuan loker ini."
                reasons.append(r)
                if log_detail:
                    logger.info("    2d. [gender_match=None, usia_match=True] %s", r)

        # 3. SKILL
        if skill_match is None:
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills, lowongan.skills, embedding_service
            )
        _, matched_skills, unmatched_skills = skill_match

        if not lowongan.skills:
            r = f"Loker {lowongan.namalowongan} tidak mencantumkan daftar skill yang dibutuhkan."
            reasons.append(r)
            if log_detail:
                logger.info("    3. [loker tidak minta skill] %s", r)
        elif not pelamar.skills:
            r = ("Pelamar belum mencantumkan skill di profil. "
                 "Penilaian skill tidak dapat dilakukan — "
                 "disarankan HR memverifikasi kemampuan saat interview.")
            reasons.append(r)
            if log_detail:
                logger.info("    3. [pelamar tidak punya skill] %s", r)
        elif matched_skills:
            skill_str = ', '.join(
                f"{loker_nm} → {pal_nm} ({ket})"
                for loker_nm, pal_nm, ket in matched_skills[:5]
            )
            r = f"Skill yang relevan dengan kebutuhan loker: {skill_str}."
            reasons.append(r)
            if log_detail:
                logger.info("    3. [matched_skills=%d item, max 5 ditampilkan] %s", len(matched_skills), r)
            if unmatched_skills:
                r2 = f"Skill loker yang belum ter-cover di profil pelamar: {', '.join(unmatched_skills[:3])}."
                reasons.append(r2)
                if log_detail:
                    logger.info("    3b. [unmatched_skills=%d item] %s", len(unmatched_skills), r2)
        else:
            r = (f"Tidak ditemukan skill pelamar yang sesuai dengan kebutuhan "
                 f"loker {lowongan.namalowongan}. "
                 f"Skill yang dibutuhkan: {', '.join(s.nama for s in lowongan.skills[:3])}.")
            reasons.append(r)
            if log_detail:
                logger.info("    3. [tidak ada skill cocok sama sekali] %s", r)

        # 4. PENDIDIKAN
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        min_edu_req   = _get_min_edu_req(lowongan)

        if not pelamar.pendidikans:
            r = "Data riwayat pendidikan pelamar tidak tersedia."
            reasons.append(r)
            if log_detail:
                logger.info("    4. [tidak ada data pendidikan] %s", r)
        else:
            edu_level   = TextPreprocessor.get_pendidikan_level(edu_tertinggi.kategori)
            jurusan     = edu_tertinggi.jurusan or ''
            jurusan_str = f" jurusan {jurusan}" if jurusan and jurusan != '-' else ''

            if min_edu_req == 0:
                r = (f"Pendidikan terakhir pelamar: {edu_tertinggi.kategori}{jurusan_str}. "
                     f"Loker ini tidak mencantumkan syarat pendidikan minimum.")
                if log_detail:
                    logger.info("    4. [loker tidak syaratkan pendidikan] %s", r)
            elif edu_level >= min_edu_req:
                r = (f"Pendidikan pelamar ({edu_tertinggi.kategori}{jurusan_str}) "
                     f"memenuhi syarat minimum loker.")
                if log_detail:
                    logger.info("    4. [level %s >= syarat %s] %s", edu_level, min_edu_req, r)
            else:
                req_label = EDU_LABEL_MAP.get(min_edu_req, str(min_edu_req))
                r = (f"Pendidikan pelamar ({edu_tertinggi.kategori}{jurusan_str}) "
                     f"berada di bawah syarat minimum loker ({req_label}). "
                     f"Pertimbangkan ini sebagai faktor seleksi awal.")
                if log_detail:
                    logger.info("    4. [level %s < syarat %s] %s", edu_level, min_edu_req, r)
            reasons.append(r)

        # 5. PENGALAMAN
        if exp_reasoning is None:
            exp_reasoning = ScoringService.compute_exp_for_reasoning(
                pelamar.pengalamans, job_vec, embedding_service
            )
        best_exp, best_sim = exp_reasoning

        total_bulan   = pelamar.total_pengalaman_bulan
        min_exp_bulan = lowongan.minimal_pengalaman_bulan or 0

        if not pelamar.pengalamans:
            if min_exp_bulan == 0:
                r = ("Pelamar belum memiliki pengalaman kerja. "
                     "Loker ini terbuka untuk fresh graduate.")
            else:
                r = (f"Pelamar belum memiliki pengalaman kerja, sementara loker ini "
                     f"mensyaratkan minimal {_bulan_to_label(min_exp_bulan)} pengalaman.")
            reasons.append(r)
            if log_detail:
                logger.info("    5. [tidak ada pengalaman] %s", r)
        else:
            bln_awal = int(getattr(best_exp, 'bulanawal', 0) or 1)
            thn_awal = int(best_exp.tahunawal)

            if best_exp.aktif == 1 or best_exp.tahunselesai is None:
                today     = date.today()
                bln_akhir = today.month
                thn_akhir = today.year
            else:
                bln_akhir = int(getattr(best_exp, 'bulanselesai', 0) or 1)
                thn_akhir = int(best_exp.tahunselesai)

            durasi_bulan = max((thn_akhir - thn_awal) * 12 + (bln_akhir - bln_awal), 0)

            if best_sim >= 0.80:
                rel_desc = "sangat relevan"
            elif best_sim >= 0.65:
                rel_desc = "relevan"
            elif best_sim >= 0.50:
                rel_desc = "cukup relevan"
            else:
                rel_desc = "kurang relevan"

            status_str = (
                "masih aktif"
                if (best_exp.aktif == 1 or best_exp.tahunselesai is None)
                else f"selesai {thn_akhir}"
            )

            r = (f"Pengalaman paling relevan: {best_exp.posisi} "
                 f"({_bulan_to_label(durasi_bulan)}, {status_str}) — {rel_desc} "
                 f"untuk posisi {lowongan.namalowongan}.")
            reasons.append(r)
            if log_detail:
                logger.info("    5. [posisi paling mirip, cosine=%.4f -> %s] %s", best_sim, rel_desc, r)

            if len(pelamar.pengalamans) > 1:
                semua_posisi = [e.posisi for e in pelamar.pengalamans]
                r2 = (f"Total pengalaman kerja: {_bulan_to_label(total_bulan)} "
                      f"dari {len(pelamar.pengalamans)} posisi "
                      f"({', '.join(semua_posisi[:3])}"
                      f"{'...' if len(semua_posisi) > 3 else ''}).")
                reasons.append(r2)
                if log_detail:
                    logger.info("    5b. [%d posisi total] %s", len(pelamar.pengalamans), r2)

            if min_exp_bulan > 0 and total_bulan < min_exp_bulan:
                r3 = (f"Total pengalaman ({_bulan_to_label(total_bulan)}) "
                      f"masih kurang dari syarat minimum loker "
                      f"({_bulan_to_label(min_exp_bulan)}).")
                reasons.append(r3)
                if log_detail:
                    logger.info("    5c. [%s bulan < syarat %s bulan] %s", total_bulan, min_exp_bulan, r3)

        # 6. BAHASA
        if biodata_flags:
            lang_missing = biodata_flags.get('lang_missing', [])
            lang_found   = biodata_flags.get('lang_found', [])

            if lang_found and not lang_missing:
                r = (f"Kemampuan bahasa yang disyaratkan loker "
                     f"({', '.join(lang_found)}) ditemukan di profil pelamar.")
                reasons.append(r)
                if log_detail:
                    logger.info("    6. [lang_found=%s] %s", lang_found, r)
            elif lang_missing:
                r = (f"Loker mensyaratkan kemampuan {', '.join(lang_missing)}, "
                     f"namun tidak ditemukan di profil pelamar. "
                     f"Verifikasi saat interview disarankan.")
                reasons.append(r)
                if log_detail:
                    logger.info("    6. [lang_missing=%s] %s", lang_missing, r)

        # 7. CATATAN DATA TIPIS
        data_tipis = []
        if not pelamar.deskripsidiri:
            data_tipis.append('deskripsi diri kosong')
        if not pelamar.skills:
            data_tipis.append('skill tidak diisi')
        elif len(pelamar.skills) == 1:
            data_tipis.append('hanya 1 skill tercantum')

        if len(data_tipis) >= 2:
            r = (f"Catatan: profil pelamar masih tipis ({', '.join(data_tipis)}). "
                 f"Skor mungkin tidak merepresentasikan kemampuan sebenarnya — "
                 f"verifikasi manual disarankan.")
            reasons.append(r)
            if log_detail:
                logger.info("    7. [data_tipis=%s] %s", data_tipis, r)

        return reasons

    # ============================================================
    # BUILD BIODATA FLAGS
    # ============================================================
    @staticmethod
    def build_biodata_flags(pelamar, lowongan, log_detail: bool = False) -> dict:
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        edu_kategori  = edu_tertinggi.kategori if edu_tertinggi else None
        skill_names   = _get_skills_text(pelamar)

        biodata = {
            'tanggallahir': getattr(pelamar, 'tanggallahir', None),
            'jeniskelamin': getattr(pelamar, 'jeniskelamin', None),
        }

        return _validator.validate(
            pelamar_biodata        = biodata,
            lowongan               = lowongan,
            pelamar_edu_kategori   = edu_kategori,
            total_pengalaman_bulan = pelamar.total_pengalaman_bulan,
            pelamar_skills_raw     = skill_names,
            log_detail             = log_detail,
        )