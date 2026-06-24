from datetime import date, datetime
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import TextPreprocessor, SKILL_THRESHOLD
from .scoring import ScoringService
from .biodata_validator import BiodataValidator

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
    ) -> list:
        tags = []

        # TAG 1 — BIODATA
        if biodata_flags:
            gm = biodata_flags.get('gender_match')
            um = biodata_flags.get('usia_match')

            if gm is True:
                tags.append({
                    'type': 'success',
                    'text': biodata_flags.get('gender_note', 'Gender sesuai')
                })
            elif gm is False:
                tags.append({
                    'type': 'danger',
                    'text': biodata_flags.get('gender_note', 'Gender tidak sesuai')
                })

            if um is True:
                usia = biodata_flags.get('usia')
                tags.append({
                    'type': 'success',
                    'text': f'Usia {usia} tahun — sesuai syarat'
                })
            elif um is False:
                tags.append({
                    'type': 'danger',
                    'text': biodata_flags.get('usia_note', 'Usia tidak memenuhi syarat')
                })

        # TAG 2 — SKILL (gunakan skill_match yang sudah dihitung)
        if skill_match is None:
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills, lowongan.skills, embedding_service
            )
        _, matched_skills, _ = skill_match

        if not lowongan.skills:
            pass
        elif not pelamar.skills:
            tags.append({'type': 'warning', 'text': 'Belum ada skill terdaftar di profil'})
        elif matched_skills:
            for loker_nama, _, ket in matched_skills[:3]:
                tags.append({'type': 'success', 'text': f'{loker_nama} ({ket})'})
        else:
            tags.append({'type': 'danger', 'text': 'Skill belum relevan dengan kebutuhan loker'})

        # TAG 3 — BAHASA
        if biodata_flags:
            lang_missing = biodata_flags.get('lang_missing', [])
            lang_found   = biodata_flags.get('lang_found', [])
            if lang_missing:
                tags.append({
                    'type': 'danger',
                    'text': f'Tidak ada bukti kemampuan: {", ".join(lang_missing)}'
                })
            elif lang_found:
                tags.append({
                    'type': 'success',
                    'text': f'Bahasa {", ".join(lang_found)} tersedia'
                })

        # TAG 4 — PENDIDIKAN
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        min_edu_req   = _get_min_edu_req(lowongan)

        if not pelamar.pendidikans:
            tags.append({'type': 'warning', 'text': 'Data pendidikan tidak tersedia'})
        else:
            edu_level = TextPreprocessor.get_pendidikan_level(edu_tertinggi.kategori)
            edu_label = edu_tertinggi.kategori

            if min_edu_req == 0:
                tags.append({'type': 'info', 'text': f'Pendidikan: {edu_label}'})
            elif edu_level >= min_edu_req:
                tags.append({'type': 'success', 'text': f'Pendidikan {edu_label} — memenuhi syarat'})
            else:
                tags.append({'type': 'danger', 'text': f'Pendidikan {edu_label} — di bawah syarat minimum'})

        # TAG 5 — PENGALAMAN (gunakan exp_reasoning yang sudah dihitung)
        if exp_reasoning is None:
            exp_reasoning = ScoringService.compute_exp_for_reasoning(
                pelamar.pengalamans, job_vec, embedding_service
            )
        best_exp, best_sim = exp_reasoning

        total_bulan   = pelamar.total_pengalaman_bulan
        min_exp_bulan = lowongan.minimal_pengalaman_bulan or 0

        if not pelamar.pengalamans:
            if min_exp_bulan > 0:
                tags.append({
                    'type': 'danger',
                    'text': f'Tidak ada pengalaman (loker min. {_bulan_to_label(min_exp_bulan)})'
                })
            else:
                tags.append({'type': 'warning', 'text': 'Belum ada pengalaman kerja'})
        else:
            if best_sim >= 0.55:
                rel_label, rel_type = 'sangat relevan', 'success'
            elif best_sim >= 0.40:
                rel_label, rel_type = 'cukup relevan', 'success'
            else:
                rel_label, rel_type = 'kurang relevan', 'warning'

            dur_label = _bulan_to_label(total_bulan)

            if min_exp_bulan > 0 and total_bulan < min_exp_bulan:
                tags.append({
                    'type': 'warning',
                    'text': f'Pengalaman {dur_label} — kurang dari syarat {_bulan_to_label(min_exp_bulan)}'
                })
            else:
                tags.append({'type': rel_type, 'text': f'Pengalaman {dur_label} ({rel_label})'})

        return tags    
   
    @staticmethod
    def generate_tags_rekomendasi(
        pelamar,
        lowongan,
        job_vec,
        embedding_service,
        biodata_flags: dict = None,
        skill_match: tuple = None,
        exp_reasoning: tuple = None,
    ) -> list:
        tags = []

        # TAG 1 — BIODATA
        if biodata_flags:
            gm = biodata_flags.get('gender_match')
            um = biodata_flags.get('usia_match')

            if gm is True:
                tags.append({'type': 'success', 'text': 'Gender sesuai'})
            elif gm is False:
                tags.append({'type': 'danger', 'text': 'Gender tidak sesuai'})

            if um is True:
                usia = biodata_flags.get('usia')
                tags.append({'type': 'success', 'text': f'Usia {usia} thn sesuai'})
            elif um is False:
                tags.append({'type': 'danger', 'text': 'Usia tidak memenuhi syarat'})

        # TAG 2 — SKILL
        if skill_match is None:
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills, lowongan.skills, embedding_service
            )
        _, matched_skills, _ = skill_match

        if not lowongan.skills:
            pass
        elif not pelamar.skills:
            tags.append({'type': 'warning', 'text': 'Belum ada skill'})
        elif matched_skills:
            tags.append({'type': 'success', 'text': 'Skill relevan'})
        else:
            tags.append({'type': 'warning', 'text': 'Skill kurang sesuai'})

        # TAG 3 — BAHASA
        if biodata_flags:
            lang_missing = biodata_flags.get('lang_missing', [])
            lang_found   = biodata_flags.get('lang_found', [])
            if lang_missing:
                tags.append({'type': 'danger', 'text': f'Perlu {", ".join(lang_missing)}'})
            elif lang_found:
                tags.append({'type': 'success', 'text': f'Bahasa {", ".join(lang_found)} ✓'})

        # TAG 4 — PENDIDIKAN
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        min_edu_req   = _get_min_edu_req(lowongan)

        if not pelamar.pendidikans:
            tags.append({'type': 'warning', 'text': 'Data pendidikan kosong'})
        else:
            edu_level = TextPreprocessor.get_pendidikan_level(edu_tertinggi.kategori)
            if min_edu_req == 0:
                tags.append({'type': 'info', 'text': 'Pendidikan tidak disyaratkan'})
            elif edu_level >= min_edu_req:
                tags.append({'type': 'success', 'text': f'Pendidikan {edu_tertinggi.kategori} ✓'})
            else:
                req_label = EDU_LABEL_MAP.get(min_edu_req, str(min_edu_req))
                tags.append({'type': 'danger', 'text': f'Pendidikan Anda di bawah syarat {req_label}'})

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
            else:
                tags.append({
                    'type': 'danger',
                    'text': f'Perlu pengalaman min. {_bulan_to_label(min_exp_bulan)}'
                })
        else:
            if min_exp_bulan == 0:
                tags.append({'type': 'success', 'text': 'Fresh graduate welcome'})
            elif total_bulan < min_exp_bulan:
                tags.append({
                    'type': 'warning',
                    'text': f'Pengalaman {_bulan_to_label(total_bulan)} (min. {_bulan_to_label(min_exp_bulan)})'
                })
            elif best_sim >= 0.45:
                tags.append({'type': 'success', 'text': 'Pengalaman relevan'})
            elif best_sim >= 0.30:
                tags.append({'type': 'warning', 'text': 'Pengalaman cukup relevan'})
            else:
                tags.append({'type': 'warning', 'text': 'Pengalaman kurang relevan'})

        return tags
   
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
    ) -> list:
        reasons = []

        # 1. SEMANTIC
        sem = scores.get('semantic', 0)
        if sem >= 0.55:
            reasons.append(
                f"Profil pelamar secara keseluruhan sangat sesuai dengan "
                f"deskripsi lowongan {lowongan.namalowongan} "
                f"(kecocokan semantik {round(sem * 100)}%)."
            )
        elif sem >= 0.40:
            reasons.append(
                f"Profil pelamar cukup sesuai dengan deskripsi lowongan "
                f"{lowongan.namalowongan} "
                f"(kecocokan semantik {round(sem * 100)}%)."
            )
        else:
            reasons.append(
                f"Kecocokan profil pelamar dengan deskripsi lowongan "
                f"{lowongan.namalowongan} masih rendah "
                f"(kecocokan semantik {round(sem * 100)}%). "
                f"Profil pelamar mungkin perlu dilengkapi lebih lanjut."
            )

        # 2. BIODATA
        if biodata_flags:
            gm = biodata_flags.get('gender_match')
            um = biodata_flags.get('usia_match')

            if gm is False:
                reasons.append(biodata_flags['gender_note'] + '.')
            if um is False:
                reasons.append(biodata_flags['usia_note'] + '.')
            if gm is True and um is True:
                usia       = biodata_flags.get('usia')
                gender_val = getattr(pelamar, 'jeniskelamin', '') or ''
                reasons.append(
                    f"Pelamar memenuhi syarat biodata loker: "
                    f"jenis kelamin {gender_val} dan usia {usia} tahun sesuai ketentuan."
                )
            elif gm is None and um is True:
                usia = biodata_flags.get('usia')
                reasons.append(
                    f"Usia pelamar ({usia} tahun) sesuai dengan ketentuan loker ini."
                )

        # 3. SKILL (gunakan skill_match yang sudah dihitung)
        if skill_match is None:
            skill_match = ScoringService.compute_skill_match(
                pelamar.skills, lowongan.skills, embedding_service
            )
        _, matched_skills, unmatched_skills = skill_match

        if not lowongan.skills:
            reasons.append(
                f"Loker {lowongan.namalowongan} tidak mencantumkan daftar skill "
                f"yang dibutuhkan. Penilaian skill dilakukan secara semantik."
            )
        elif not pelamar.skills:
            reasons.append(
                "Pelamar belum mencantumkan skill di profil. "
                "Penilaian skill tidak dapat dilakukan — "
                "disarankan HR memverifikasi kemampuan saat interview."
            )
        elif matched_skills:
            skill_str = ', '.join(
                f"{loker_nm} → {pal_nm} ({ket})"
                for loker_nm, pal_nm, ket in matched_skills[:5]
            )
            reasons.append(f"Skill yang relevan dengan kebutuhan loker: {skill_str}.")
            if unmatched_skills:
                reasons.append(
                    f"Skill loker yang belum ter-cover di profil pelamar: "
                    f"{', '.join(unmatched_skills[:3])}."
                )
        else:
            reasons.append(
                f"Tidak ditemukan skill pelamar yang cocok dengan kebutuhan "
                f"loker {lowongan.namalowongan}. "
                f"Skill yang dibutuhkan: {', '.join(s.nama for s in lowongan.skills[:3])}."
            )

        # 4. PENDIDIKAN
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        min_edu_req   = _get_min_edu_req(lowongan)

        if not pelamar.pendidikans:
            reasons.append("Data riwayat pendidikan pelamar tidak tersedia.")
        else:
            edu_level   = TextPreprocessor.get_pendidikan_level(edu_tertinggi.kategori)
            jurusan     = edu_tertinggi.jurusan or ''
            jurusan_str = f" jurusan {jurusan}" if jurusan and jurusan != '-' else ''

            if min_edu_req == 0:
                reasons.append(
                    f"Pendidikan terakhir pelamar: {edu_tertinggi.kategori}{jurusan_str}. "
                    f"Loker ini tidak mencantumkan syarat pendidikan minimum."
                )
            elif edu_level >= min_edu_req:
                reasons.append(
                    f"Pendidikan pelamar ({edu_tertinggi.kategori}{jurusan_str}) "
                    f"memenuhi syarat minimum loker."
                )
            else:
                req_label = EDU_LABEL_MAP.get(min_edu_req, str(min_edu_req))
                reasons.append(
                    f"Pendidikan pelamar ({edu_tertinggi.kategori}{jurusan_str}) "
                    f"berada di bawah syarat minimum loker ({req_label}). "
                    f"Pertimbangkan ini sebagai faktor seleksi awal."
                )

        # 5. PENGALAMAN (gunakan exp_reasoning yang sudah dihitung)
        if exp_reasoning is None:
            exp_reasoning = ScoringService.compute_exp_for_reasoning(
                pelamar.pengalamans, job_vec, embedding_service
            )
        best_exp, best_sim = exp_reasoning

        total_bulan   = pelamar.total_pengalaman_bulan
        min_exp_bulan = lowongan.minimal_pengalaman_bulan or 0

        if not pelamar.pengalamans:
            if min_exp_bulan == 0:
                reasons.append(
                    "Pelamar belum memiliki pengalaman kerja. "
                    "Loker ini terbuka untuk fresh graduate."
                )
            else:
                reasons.append(
                    f"Pelamar belum memiliki pengalaman kerja, sementara loker ini "
                    f"mensyaratkan minimal {_bulan_to_label(min_exp_bulan)} pengalaman."
                )
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

            durasi_bulan = max(
                (thn_akhir - thn_awal) * 12 + (bln_akhir - bln_awal), 0
            )

            if best_sim >= 0.55:
                rel_desc = "sangat relevan"
            elif best_sim >= 0.40:
                rel_desc = "cukup relevan"
            else:
                rel_desc = "kurang relevan secara langsung"

            status_str = (
                "masih aktif"
                if (best_exp.aktif == 1 or best_exp.tahunselesai is None)
                else f"selesai {thn_akhir}"
            )

            reasons.append(
                f"Pengalaman paling relevan: {best_exp.posisi} "
                f"({_bulan_to_label(durasi_bulan)}, {status_str}) — {rel_desc} "
                f"untuk posisi {lowongan.namalowongan}."
            )

            if len(pelamar.pengalamans) > 1:
                semua_posisi = [e.posisi for e in pelamar.pengalamans]
                reasons.append(
                    f"Total pengalaman kerja: {_bulan_to_label(total_bulan)} "
                    f"dari {len(pelamar.pengalamans)} posisi "
                    f"({', '.join(semua_posisi[:3])}"
                    f"{'...' if len(semua_posisi) > 3 else ''})."
                )

            if min_exp_bulan > 0 and total_bulan < min_exp_bulan:
                reasons.append(
                    f"Total pengalaman ({_bulan_to_label(total_bulan)}) "
                    f"masih kurang dari syarat minimum loker "
                    f"({_bulan_to_label(min_exp_bulan)})."
                )

        # 6. BAHASA
        if biodata_flags:
            lang_missing = biodata_flags.get('lang_missing', [])
            lang_found   = biodata_flags.get('lang_found', [])

            if lang_found and not lang_missing:
                reasons.append(
                    f"Kemampuan bahasa yang disyaratkan loker "
                    f"({', '.join(lang_found)}) ditemukan di profil pelamar."
                )
            elif lang_missing:
                reasons.append(
                    f"Loker mensyaratkan kemampuan {', '.join(lang_missing)}, "
                    f"namun tidak ditemukan di profil pelamar. "
                    f"Verifikasi saat interview disarankan."
                )

        # 7. CATATAN DATA TIPIS
        data_tipis = []
        if not pelamar.deskripsidiri:
            data_tipis.append('deskripsi diri kosong')
        if not pelamar.skills:
            data_tipis.append('skill tidak diisi')
        elif len(pelamar.skills) == 1:
            data_tipis.append('hanya 1 skill tercantum')

        if len(data_tipis) >= 2:
            reasons.append(
                f"Catatan: profil pelamar masih tipis ({', '.join(data_tipis)}). "
                f"Skor mungkin tidak merepresentasikan kemampuan sebenarnya — "
                f"verifikasi manual disarankan."
            )

        return reasons



    @staticmethod
    def build_biodata_flags(pelamar, lowongan) -> dict:
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
        )