from datetime import date, datetime
from .preprocess import TextPreprocessor


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


def _normalize_gender(raw: str) -> str | None:
    if not raw:
        return None
    r = str(raw).strip().lower()
    if r in ('perempuan', 'wanita', 'p', 'f'):
        return 'Perempuan'
    if r in ('laki-laki', 'laki laki', 'pria', 'l', 'm'):
        return 'Laki-laki'
    return None


# Keyword bahasa yang dicek dari nama skill loker & skill pelamar
_LANG_KEYWORDS = {
    'Mandarin': ['mandarin', 'tiongkok', 'chinese'],
    'Inggris':  ['inggris', 'english'],
}


class BiodataValidator:

    def validate(
        self,
        pelamar_biodata: dict,
        lowongan,
        pelamar_edu_kategori: str = None,
        total_pengalaman_bulan: int = 0,
        pelamar_skills_raw: list = None,
    ) -> dict:
        result = {}

        # Ambil syarat dari object lowongan langsung
        req_gender    = None if lowongan.preferensi_gender == 'Semua' \
                        else lowongan.preferensi_gender
        min_usia      = lowongan.usia_min if lowongan.usia_min and lowongan.usia_min > 0 else None
        max_usia      = lowongan.usia_max if lowongan.usia_max and lowongan.usia_max > 0 else None
        min_edu_kode  = lowongan.minimal_pendidikan.kode \
                        if lowongan.minimal_pendidikan else 0
        min_edu_req   = TextPreprocessor.get_pendidikan_level_from_kode(min_edu_kode)
        min_exp_bulan = lowongan.minimal_pengalaman_bulan or 0

        # --------------------------------------------------------
        # 1. USIA
        # --------------------------------------------------------
        usia = _hitung_usia(pelamar_biodata.get('tanggallahir'))
        result['usia'] = usia

        if usia is None:
            result['usia_match'] = None
            result['usia_note']  = 'Data tanggal lahir tidak tersedia'

        elif min_usia is None and max_usia is None:
            result['usia_match'] = None
            result['usia_note']  = 'Loker tidak mencantumkan syarat usia'

        else:
            too_young = min_usia is not None and usia < min_usia
            too_old   = max_usia is not None and usia > max_usia

            if too_young:
                result['usia_match'] = False
                result['usia_note']  = (
                    f'Usia pelamar {usia} tahun di bawah batas minimum '
                    f'{min_usia} tahun yang disyaratkan loker'
                )
            elif too_old:
                result['usia_match'] = False
                result['usia_note']  = (
                    f'Usia pelamar {usia} tahun melebihi batas maksimal '
                    f'{max_usia} tahun yang disyaratkan loker'
                )
            else:
                result['usia_match'] = True
                if min_usia and max_usia:
                    range_str = f' (syarat: {min_usia}–{max_usia} tahun)'
                elif max_usia:
                    range_str = f' (syarat: maksimal {max_usia} tahun)'
                else:
                    range_str = f' (syarat: minimal {min_usia} tahun)'
                result['usia_note'] = (
                    f'Usia pelamar {usia} tahun memenuhi syarat usia loker{range_str}'
                )

        # --------------------------------------------------------
        # 2. GENDER
        # --------------------------------------------------------
        pelamar_gender = _normalize_gender(
            pelamar_biodata.get('jeniskelamin', '')
        )

        if req_gender is None:
            result['gender_match'] = None
            result['gender_note']  = 'Loker tidak mencantumkan syarat gender'

        elif pelamar_gender is None:
            result['gender_match'] = None
            result['gender_note']  = 'Data gender pelamar tidak tersedia'

        elif pelamar_gender == req_gender:
            result['gender_match'] = True
            result['gender_note']  = (
                f'Jenis kelamin pelamar ({pelamar_gender}) '
                f'sesuai dengan syarat loker'
            )
        else:
            result['gender_match'] = False
            result['gender_note']  = (
                f'Loker mensyaratkan {req_gender}, '
                f'pelamar berjenis kelamin {pelamar_gender}'
            )

        # --------------------------------------------------------
        # 3. PENDIDIKAN
        # --------------------------------------------------------
        EDU_LABEL_MAP = {
            9: 'S3', 8: 'S2', 7: 'D4/S1', 6: 'D3',
            5: 'D2', 4: 'D1', 3: 'SMA/SMK', 2: 'SMP', 1: 'SD'
        }

        if min_edu_req == 0:
            result['edu_gap']  = False
            result['edu_note'] = 'Loker tidak mencantumkan syarat pendidikan minimum'

        elif pelamar_edu_kategori is None:
            result['edu_gap']  = False
            result['edu_note'] = 'Data pendidikan pelamar tidak tersedia'

        else:
            pelamar_edu_level = TextPreprocessor.get_pendidikan_level(
                pelamar_edu_kategori
            )
            req_label = EDU_LABEL_MAP.get(min_edu_req, str(min_edu_req))

            if pelamar_edu_level >= min_edu_req:
                result['edu_gap']  = False
                result['edu_note'] = (
                    f'Pendidikan pelamar ({pelamar_edu_kategori}) memenuhi '
                    f'syarat minimum loker ({req_label})'
                )
            else:
                result['edu_gap']  = True
                result['edu_note'] = (
                    f'Pendidikan pelamar ({pelamar_edu_kategori}) di bawah '
                    f'syarat minimum loker ({req_label})'
                )

        # --------------------------------------------------------
        # 4. PENGALAMAN (dalam bulan)
        # --------------------------------------------------------
        if min_exp_bulan == 0:
            result['exp_gap']  = False
            result['exp_note'] = 'Loker terbuka untuk fresh graduate'

        else:
            if total_pengalaman_bulan >= min_exp_bulan:
                result['exp_gap']  = False
                result['exp_note'] = (
                    f'Total pengalaman pelamar ({total_pengalaman_bulan} bulan) '
                    f'memenuhi syarat minimum loker ({min_exp_bulan} bulan)'
                )
            else:
                result['exp_gap']  = True
                result['exp_note'] = (
                    f'Total pengalaman pelamar ({total_pengalaman_bulan} bulan) '
                    f'kurang dari syarat minimum loker ({min_exp_bulan} bulan)'
                )

        # --------------------------------------------------------
        # 5. BAHASA — deteksi dari nama skill loker vs skill pelamar
        # --------------------------------------------------------
        skill_loker_text   = ' '.join(
            s.nama.lower() for s in (lowongan.skills or [])
        )
        skill_pelamar_text = ' '.join(
            str(s).lower() for s in (pelamar_skills_raw or [])
        )

        lang_found   = []
        lang_missing = []

        for lang, keywords in _LANG_KEYWORDS.items():
            loker_butuh = any(kw in skill_loker_text for kw in keywords)
            if loker_butuh:
                pelamar_punya = any(kw in skill_pelamar_text for kw in keywords)
                if pelamar_punya:
                    lang_found.append(lang)
                else:
                    lang_missing.append(lang)

        result['lang_found']   = lang_found
        result['lang_missing'] = lang_missing

        if not lang_found and not lang_missing:
            result['lang_note'] = 'Loker tidak mensyaratkan kemampuan bahasa khusus'
        elif not lang_missing:
            result['lang_note'] = (
                f'Pelamar memiliki kemampuan bahasa yang dibutuhkan loker: '
                f'{", ".join(lang_found)}'
            )
        else:
            result['lang_note'] = (
                f'Bahasa yang dibutuhkan loker namun tidak ditemukan '
                f'di profil pelamar: {", ".join(lang_missing)}'
            )

        return result