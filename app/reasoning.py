from datetime import date, datetime
from sklearn.metrics.pairwise import cosine_similarity

from .scoring import SKILL_THRESHOLD, PROFICIENCY_WEIGHT, _edu_level_from_str
from .description_parser import DescriptionParser
from .biodata_validator import BiodataValidator

_parser    = DescriptionParser()
_validator = BiodataValidator()

CURRENT_YEAR = datetime.now().year


# HELPER INTERNAL
def _hitung_usia(tanggallahir) -> int | None:
    if tanggallahir is None:
        return None
    if isinstance(tanggallahir, str):
        try:
            tanggallahir = datetime.strptime(
                str(tanggallahir)[:10], '%Y-%m-%d'
            ).date()
        except ValueError:
            return None
    today = date.today()
    return today.year - tanggallahir.year - (
        (today.month, today.day) < (tanggallahir.month, tanggallahir.day)
    )


def _get_matched_skills(pelamar, job_vec, embedding_service) -> tuple:
    matched   = []
    unmatched = []

    for skill in (pelamar.skills or []):
        skill_text = f"memiliki keahlian {skill.namaskill}"
        skill_vec  = embedding_service.encode(skill_text)
        sim = float(cosine_similarity([skill_vec], [job_vec])[0][0])

        if sim >= SKILL_THRESHOLD:
            matched.append((skill.namaskill, skill.keterangan))
        else:
            unmatched.append(skill.namaskill)

    return matched, unmatched


def _get_edu_tertinggi(pelamar):
    # Return objek pendidikan tertinggi pelamar, atau None.
    if not pelamar.pendidikans:
        return None
    return max(pelamar.pendidikans, key=lambda p: _edu_level_from_str(p.kategori))


def _hitung_total_exp(pelamar) -> float:
    # Total tahun pengalaman kerja.
    if not pelamar.pengalamans:
        return 0.0
    total = 0
    for exp in pelamar.pengalamans:
        thn_awal  = int(exp.tahunawal)
        thn_akhir = (
            CURRENT_YEAR
            if exp.aktif == 1 or exp.tahunselesai is None
            else int(exp.tahunselesai)
        )
        thn_akhir = max(thn_awal, min(thn_akhir, CURRENT_YEAR))
        total += thn_akhir - thn_awal
    return float(total)


def _get_most_relevant_exp(pelamar, job_vec, embedding_service):
    # Return pengalaman paling relevan (objek + similarity score). Return (None, 0.0) jika tidak ada pengalaman.
    if not pelamar.pengalamans:
        return None, 0.0

    best_exp = None
    best_sim = -1.0

    for exp in pelamar.pengalamans:
        posisi_text = f"pengalaman kerja sebagai {exp.posisi}"
        posisi_vec  = embedding_service.encode(posisi_text)
        sim = float(cosine_similarity([posisi_vec], [job_vec])[0][0])
        if sim > best_sim:
            best_sim = sim
            best_exp = exp

    return best_exp, best_sim


def _get_skills_text(pelamar) -> list:
    # Return list nama skill sebagai string untuk cek bahasa.
    return [s.namaskill for s in (pelamar.skills or [])]


# REASONING SERVICE
class ReasoningService:
    # GENERATE TAGS
    @staticmethod
    def generate_tags(
        pelamar,
        job_vec,
        embedding_service,
        final_score: float,
        parsed_desc: dict = None,
        biodata_flags: dict = None,
    ) -> list:
        tags = []
        req  = (parsed_desc or {}).get('hard_requirements', {})

        # TAG 1: BIODATA — Gender & Usia (jika loker mensyaratkan)
        if biodata_flags:
            # Gender
            gm = biodata_flags.get('gender_match')
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
            # None = tidak disyaratkan → tidak ada tag gender

            # Usia
            um = biodata_flags.get('usia_match')
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

        # TAG 2: SKILL
        matched_skills, _ = _get_matched_skills(pelamar, job_vec, embedding_service)

        if not pelamar.skills:
            tags.append({
                'type': 'warning',
                'text': 'Belum ada skill terdaftar di profil'
            })
        elif matched_skills:
            # Tampilkan max 3 skill yang match
            for nama, level in matched_skills[:3]:
                tags.append({
                    'type': 'success',
                    'text': f'{nama} ({level})'
                })
        else:
            tags.append({
                'type': 'danger',
                'text': 'Skill belum relevan dengan kebutuhan loker'
            })

        # TAG 3: BAHASA (jika loker mensyaratkan)
        if biodata_flags:
            lang_missing = biodata_flags.get('lang_missing', [])
            if lang_missing:
                tags.append({
                    'type': 'danger',
                    'text': f'Tidak ada bukti kemampuan: {", ".join(lang_missing)}'
                })
            elif req.get('required_langs'):
                # Ada syarat bahasa dan semua terpenuhi
                tags.append({
                    'type': 'success',
                    'text': f'Bahasa {", ".join(req["required_langs"])} tersedia'
                })

        # TAG 4: PENDIDIKAN — kontekstual terhadap syarat loker
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        min_edu_req   = req.get('min_edu', 0)

        if not pelamar.pendidikans:
            tags.append({
                'type': 'warning',
                'text': 'Data pendidikan tidak tersedia'
            })
        else:
            edu_level    = _edu_level_from_str(edu_tertinggi.kategori)
            edu_label    = edu_tertinggi.kategori

            if min_edu_req == 0:
                # Loker tidak mensyaratkan pendidikan → info saja
                tags.append({
                    'type': 'info',
                    'text': f'Pendidikan: {edu_label}'
                })
            elif edu_level >= min_edu_req:
                tags.append({
                    'type': 'success',
                    'text': f'Pendidikan {edu_label} — memenuhi syarat'
                })
            else:
                tags.append({
                    'type': 'danger',
                    'text': f'Pendidikan {edu_label} — di bawah syarat minimum'
                })

        # TAG 5: PENGALAMAN — relevansi + durasi, bukan hanya durasi
        best_exp, best_sim = _get_most_relevant_exp(
            pelamar, job_vec, embedding_service
        )
        total_exp    = _hitung_total_exp(pelamar)
        min_exp_req  = req.get('min_exp_years')

        if not pelamar.pengalamans:
            if min_exp_req is not None and min_exp_req > 0:
                tags.append({
                    'type': 'danger',
                    'text': f'Tidak ada pengalaman (loker min. {min_exp_req} tahun)'
                })
            else:
                tags.append({
                    'type': 'warning',
                    'text': 'Belum ada pengalaman kerja'
                })
        else:
            # Tentukan label pengalaman berdasarkan relevansi
            if best_sim >= 0.55:
                rel_label = 'sangat relevan'
                rel_type  = 'success'
            elif best_sim >= 0.40:
                rel_label = 'cukup relevan'
                rel_type  = 'success'
            else:
                rel_label = 'kurang relevan'
                rel_type  = 'warning'

            # Durasi: kalau 0 tapi ada pengalaman → "< 1 tahun"
            if total_exp == 0:
                dur_label = '< 1 tahun'
            elif total_exp == 1:
                dur_label = '±1 tahun'
            else:
                dur_label = f'±{int(total_exp)} tahun'

            # Cek gap terhadap syarat minimum
            if min_exp_req and total_exp < min_exp_req:
                tags.append({
                    'type': 'warning',
                    'text': (
                        f'Pengalaman {dur_label} — '
                        f'kurang dari syarat {min_exp_req} tahun'
                    )
                })
            else:
                tags.append({
                    'type': rel_type,
                    'text': f'Pengalaman {dur_label} ({rel_label})'
                })

        return tags

    # GENERATE REASONS
    @staticmethod
    def generate_reasons(
        pelamar,
        lowongan,
        job_vec,
        embedding_service,
        scores: dict,
        parsed_desc: dict = None,
        biodata_flags: dict = None,
    ) -> list:
        reasons = []
        req     = (parsed_desc or {}).get('hard_requirements', {})
        nice    = (parsed_desc or {}).get('nice_to_have', [])

        # 1. SEMANTIC — kecocokan keseluruhan
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

        # 2. BIODATA — gender & usia
        if biodata_flags:
            gm = biodata_flags.get('gender_match')
            um = biodata_flags.get('usia_match')

            if gm is False:
                reasons.append(biodata_flags['gender_note'] + '.')
            if um is False:
                reasons.append(biodata_flags['usia_note'] + '.')
            if gm is True and um is True:
                usia = biodata_flags.get('usia')
                gender_val = getattr(pelamar, 'jeniskelamin', '') or ''
                reasons.append(
                    f"Pelamar memenuhi syarat biodata loker: "
                    f"jenis kelamin {gender_val} dan usia {usia} tahun sesuai ketentuan."
                )
            elif gm is True and um is None:
                pass  # tidak perlu disebutkan jika usia tidak disyaratkan
            elif gm is None and um is True:
                usia = biodata_flags.get('usia')
                reasons.append(
                    f"Usia pelamar ({usia} tahun) sesuai dengan "
                    f"ketentuan loker ini."
                )

        # 3. SKILL
        matched_skills, unmatched_skills = _get_matched_skills(
            pelamar, job_vec, embedding_service
        )

        if not pelamar.skills:
            reasons.append(
                "Pelamar belum mencantumkan skill di profil. "
                "Penilaian skill tidak dapat dilakukan — "
                "disarankan HR memverifikasi kemampuan saat interview."
            )
        elif matched_skills:
            skill_str = ', '.join(
                f"{nm} ({lv})" for nm, lv in matched_skills[:5]
            )
            reasons.append(
                f"Skill yang relevan dengan loker: {skill_str}."
            )
            if unmatched_skills:
                unmatched_str = ', '.join(unmatched_skills[:3])
                reasons.append(
                    f"Skill lain yang tercantum namun kurang relevan "
                    f"dengan kebutuhan loker ini: {unmatched_str}."
                )
        else:
            reasons.append(
                "Tidak ditemukan skill yang langsung relevan dengan "
                f"kebutuhan loker {lowongan.namalowongan}. "
                "Skill yang tercantum di profil belum mencerminkan "
                "kompetensi utama yang dibutuhkan."
            )

        # 4. PENDIDIKAN — kontekstual
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        min_edu_req   = req.get('min_edu', 0)

        if not pelamar.pendidikans:
            reasons.append("Data riwayat pendidikan pelamar tidak tersedia.")

        else:
            edu_level = _edu_level_from_str(edu_tertinggi.kategori)
            jurusan   = edu_tertinggi.jurusan or ''
            jurusan_str = f" jurusan {jurusan}" if jurusan and jurusan != '-' else ''

            if min_edu_req == 0:
                # Loker tidak mensyaratkan → sebutkan saja tanpa judgement
                reasons.append(
                    f"Pendidikan terakhir pelamar: "
                    f"{edu_tertinggi.kategori}{jurusan_str}. "
                    f"Loker ini tidak mencantumkan syarat pendidikan minimum."
                )
            elif edu_level >= min_edu_req:
                reasons.append(
                    f"Pendidikan pelamar ({edu_tertinggi.kategori}"
                    f"{jurusan_str}) memenuhi syarat minimum loker."
                )
            else:
                # Gap pendidikan — jelaskan dengan jelas
                edu_label_map = {
                    10: 'S3', 8: 'S2', 6: 'S1/D4', 5: 'D3',
                    4: 'D1/D2', 3: 'SMA/SMK', 2: 'SMP', 1: 'SD'
                }
                req_label = edu_label_map.get(min_edu_req, str(min_edu_req))
                reasons.append(
                    f"Pendidikan pelamar ({edu_tertinggi.kategori}"
                    f"{jurusan_str}) berada di bawah syarat minimum "
                    f"loker ({req_label}). "
                    f"Pertimbangkan ini sebagai faktor seleksi awal."
                )

        # 5. PENGALAMAN — relevansi + durasi + recency
        best_exp, best_sim = _get_most_relevant_exp(
            pelamar, job_vec, embedding_service
        )
        total_exp   = _hitung_total_exp(pelamar)
        min_exp_req = req.get('min_exp_years')

        if not pelamar.pengalamans:
            if min_exp_req == 0:
                reasons.append(
                    "Pelamar belum memiliki pengalaman kerja. "
                    "Loker ini terbuka untuk fresh graduate."
                )
            elif min_exp_req and min_exp_req > 0:
                reasons.append(
                    f"Pelamar belum memiliki pengalaman kerja, "
                    f"sementara loker ini mensyaratkan minimal "
                    f"{min_exp_req} tahun pengalaman."
                )
            else:
                reasons.append(
                    "Pelamar belum memiliki pengalaman kerja."
                )
        else:
            # Pengalaman paling relevan
            posisi_relevan = best_exp.posisi
            thn_awal_rel   = int(best_exp.tahunawal)
            thn_akhir_rel  = (
                CURRENT_YEAR
                if best_exp.aktif == 1 or best_exp.tahunselesai is None
                else int(best_exp.tahunselesai)
            )
            thn_akhir_rel  = max(thn_awal_rel, min(thn_akhir_rel, CURRENT_YEAR))
            durasi_rel     = thn_akhir_rel - thn_awal_rel

            # Label relevansi
            if best_sim >= 0.55:
                rel_desc = "sangat relevan"
            elif best_sim >= 0.40:
                rel_desc = "cukup relevan"
            else:
                rel_desc = "kurang relevan secara langsung"

            # Label durasi
            if durasi_rel == 0:
                dur_str = "kurang dari 1 tahun"
            elif durasi_rel == 1:
                dur_str = "±1 tahun"
            else:
                dur_str = f"±{durasi_rel} tahun"

            # Status (masih aktif atau sudah selesai)
            status_str = (
                "masih aktif"
                if (best_exp.aktif == 1 or best_exp.tahunselesai is None)
                else f"selesai {thn_akhir_rel}"
            )

            reasons.append(
                f"Pengalaman paling relevan: {posisi_relevan} "
                f"({dur_str}, {status_str}) — {rel_desc} "
                f"untuk posisi {lowongan.namalowongan}."
            )

            # Jika ada lebih dari satu pengalaman, sebut total
            if len(pelamar.pengalamans) > 1:
                semua_posisi = [e.posisi for e in pelamar.pengalamans]
                reasons.append(
                    f"Total pengalaman kerja: ±{int(total_exp)} tahun "
                    f"dari {len(pelamar.pengalamans)} posisi "
                    f"({', '.join(semua_posisi[:3])}"
                    f"{'...' if len(semua_posisi) > 3 else ''})."
                )

            # Gap terhadap syarat minimum
            if min_exp_req and total_exp < min_exp_req:
                reasons.append(
                    f"Total pengalaman ({int(total_exp)} tahun) "
                    f"masih kurang dari syarat minimum loker "
                    f"({min_exp_req} tahun)."
                )

        # 6. BAHASA
        if biodata_flags:
            lang_missing = biodata_flags.get('lang_missing', [])
            req_langs    = req.get('required_langs', [])

            if req_langs and not lang_missing:
                reasons.append(
                    f"Kemampuan bahasa yang disyaratkan loker "
                    f"({', '.join(req_langs)}) "
                    f"ditemukan di profil pelamar."
                )
            elif lang_missing:
                reasons.append(
                    f"Loker mensyaratkan kemampuan "
                    f"{', '.join(lang_missing)}, "
                    f"namun tidak ditemukan di profil pelamar. "
                    f"Verifikasi saat interview disarankan."
                )

        # 7. NILAI TAMBAH (nice_to_have dari parser)
        if nice and matched_skills:
            # Cek apakah ada skill pelamar yang relate ke nice-to-have
            # (heuristic: kalau skill match dan ada nice-to-have → sebut)
            nice_preview = nice[0][:80] if nice else ''
            if nice_preview:
                reasons.append(
                    f"Catatan nilai tambah yang relevan dari loker: "
                    f'"{nice_preview}{"..." if len(nice[0]) > 80 else ""}"'
                )

        # 8. CATATAN DATA TIPIS
        data_tipis = []
        if not pelamar.deskripsidiri:
            data_tipis.append('deskripsi diri kosong')
        if not pelamar.skills:
            data_tipis.append('skill tidak diisi')
        if len(pelamar.skills or []) == 1:
            data_tipis.append('hanya 1 skill tercantum')

        if len(data_tipis) >= 2:
            reasons.append(
                f"Catatan: profil pelamar masih tipis "
                f"({', '.join(data_tipis)}). "
                f"Skor mungkin tidak merepresentasikan kemampuan "
                f"sebenarnya — verifikasi manual disarankan."
            )

        return reasons

    # PARSE DESC — helper untuk dipanggil dari RankerService
    @staticmethod
    def parse_lowongan(lowongan) -> dict:
        """
        Parse deskripsi lowongan dan return dict terstruktur.
        Dipanggil sekali per lowongan di RankerService.
        """
        return _parser.parse(lowongan.deskripsi or '')

    @staticmethod
    def build_biodata_flags(
        pelamar,
        hard_requirements: dict,
    ) -> dict:
        """
        Build biodata validation flags untuk satu pelamar.
        Dipanggil per pelamar di RankerService.
        """
        # Ambil kategori pendidikan tertinggi
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        edu_kategori  = edu_tertinggi.kategori if edu_tertinggi else None

        # Total exp
        total_exp = _hitung_total_exp(pelamar)

        # Skill names untuk cek bahasa
        skill_names = _get_skills_text(pelamar)

        # Biodata dict
        biodata = {
            'tanggallahir': getattr(pelamar, 'tanggallahir', None),
            'jeniskelamin': getattr(pelamar, 'jeniskelamin', None),
        }

        return _validator.validate(
            pelamar_biodata      = biodata,
            hard_requirements    = hard_requirements,
            pelamar_edu_kategori = edu_kategori,
            total_exp_years      = total_exp,
            pelamar_skills_raw   = skill_names,
        )
    
    @staticmethod
    def generate_tags_rekomendasi(
        pelamar,
        job_vec,
        embedding_service,
        parsed_desc: dict = None,
        biodata_flags: dict = None,
    ) -> list:
        tags = []
        req  = (parsed_desc or {}).get('hard_requirements', {})

        # TAG 1 — BIODATA (hanya kalau loker mensyaratkan)
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

        # TAG 2 — SKILL (selalu muncul, singkat)
        matched_skills, _ = _get_matched_skills(pelamar, job_vec, embedding_service)

        if not pelamar.skills:
            tags.append({'type': 'warning', 'text': 'Belum ada skill'})
        elif matched_skills:
            tags.append({'type': 'success', 'text': 'Skill relevan'})
        else:
            tags.append({'type': 'warning', 'text': 'Skill kurang sesuai'})

        # TAG 3 — BAHASA (hanya kalau loker mensyaratkan)
        if biodata_flags:
            lang_missing = biodata_flags.get('lang_missing', [])
            req_langs    = req.get('required_langs', [])
            if req_langs:
                if lang_missing:
                    tags.append({
                        'type': 'danger',
                        'text': f'Perlu {", ".join(lang_missing)}'
                    })
                else:
                    tags.append({
                        'type': 'success',
                        'text': f'Bahasa {", ".join(req_langs)} ✓'
                    })

        # TAG 4 — PENDIDIKAN (kontekstual)
        edu_tertinggi = _get_edu_tertinggi(pelamar)
        min_edu_req   = req.get('min_edu', 0)

        if not pelamar.pendidikans:
            tags.append({'type': 'warning', 'text': 'Data pendidikan kosong'})
        else:
            edu_level = _edu_level_from_str(edu_tertinggi.kategori)
            if min_edu_req == 0:
                tags.append({'type': 'info', 'text': 'Pendidikan tidak disyaratkan'})
            elif edu_level >= min_edu_req:
                tags.append({'type': 'success', 'text': f'Pendidikan {edu_tertinggi.kategori} ✓'})
            else:
                edu_label_map = {
                    10:'S3', 8:'S2', 6:'S1/D4', 5:'D3',
                    4:'D1/D2', 3:'SMA/SMK', 2:'SMP', 1:'SD'
                }
                req_label = edu_label_map.get(min_edu_req, str(min_edu_req))
                tags.append({
                    'type': 'danger',
                    'text': f'Pendidikan di bawah syarat {req_label}'
                })

        # TAG 5 — PENGALAMAN (relevansi + fresh grad check)
        min_exp_req       = req.get('min_exp_years')
        best_exp, best_sim = _get_most_relevant_exp(pelamar, job_vec, embedding_service)
        total_exp         = _hitung_total_exp(pelamar)

        if not pelamar.pengalamans:
            if min_exp_req == 0:
                tags.append({'type': 'success', 'text': 'Fresh graduate welcome'})
            elif min_exp_req and min_exp_req > 0:
                tags.append({
                    'type': 'danger',
                    'text': f'Perlu pengalaman min. {min_exp_req} thn'
                })
            else:
                tags.append({'type': 'warning', 'text': 'Belum ada pengalaman'})
        else:
            if min_exp_req == 0:
                tags.append({'type': 'success', 'text': 'Fresh graduate welcome'})
            elif min_exp_req and total_exp < min_exp_req:
                tags.append({
                    'type': 'warning',
                    'text': f'Pengalaman {int(total_exp)} thn (min. {min_exp_req} thn)'
                })
            elif best_sim >= 0.45:
                tags.append({'type': 'success', 'text': 'Pengalaman relevan'})
            elif best_sim >= 0.30:
                tags.append({'type': 'warning', 'text': 'Pengalaman cukup relevan'})
            else:
                tags.append({'type': 'warning', 'text': 'Pengalaman kurang relevan'})

        return tags
