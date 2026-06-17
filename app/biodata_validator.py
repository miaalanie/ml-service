from datetime import date, datetime

# HELPER
def _hitung_usia(tanggallahir) -> int | None:
    if tanggallahir is None:
        return None

    if isinstance(tanggallahir, str):
        try:
            tanggallahir = datetime.strptime(
                tanggallahir[:10], '%Y-%m-%d'
            ).date()
        except ValueError:
            return None

    today = date.today()
    usia = today.year - tanggallahir.year - (
        (today.month, today.day) < (tanggallahir.month, tanggallahir.day)
    )
    return usia


def _normalize_gender(raw: str) -> str | None:
    #Normalize gender string dari DB.
    if not raw:
        return None
    r = str(raw).strip().lower()
    if r in ('perempuan', 'wanita', 'p', 'f'):
        return 'Perempuan'
    if r in ('laki-laki', 'laki laki', 'pria', 'l', 'm'):
        return 'Laki-laki'
    return None


def _edu_level_from_str(kategori: str) -> int:
    # Mirror dari scoring.py — jangan ubah logikanya.
    k = str(kategori).upper().strip()
    if 'S3'  in k: return 10
    if 'S2'  in k: return 8
    if 'S1'  in k: return 6
    if 'D4'  in k: return 6
    if 'D3'  in k: return 5
    if 'D2'  in k: return 4
    if 'D1'  in k: return 4
    if 'SMK' in k: return 3
    if 'SMA' in k: return 3
    if 'SMP' in k: return 2
    if 'SD'  in k: return 1
    return 0


# BIODATA VALIDATOR
class BiodataValidator:
    # Validasi biodata pelamar terhadap hard requirements loker.
    # Output dipakai ReasoningService untuk reasoning kontekstual.
    def validate(
        self,
        pelamar_biodata: dict,
        hard_requirements: dict,
        pelamar_edu_kategori: str = None,
        total_exp_years: float = 0,
        pelamar_skills_raw: list = None,
    ) -> dict:
        req = hard_requirements
        result = {}

        # 1. USIA
        usia = _hitung_usia(pelamar_biodata.get('tanggallahir'))
        result['usia'] = usia

        if usia is None:
            result['usia_match'] = None
            result['usia_note']  = 'Data tanggal lahir tidak tersedia'

        elif req.get('min_usia') is None and req.get('max_usia') is None:
            result['usia_match'] = None
            result['usia_note']  = 'Loker tidak mencantumkan syarat usia'

        else:
            min_u = req.get('min_usia')
            max_u = req.get('max_usia')

            too_young = min_u is not None and usia < min_u
            too_old   = max_u is not None and usia > max_u

            if too_young:
                result['usia_match'] = False
                result['usia_note']  = (
                    f'Usia pelamar {usia} tahun di bawah batas minimum '
                    f'{min_u} tahun yang disyaratkan loker'
                )
            elif too_old:
                result['usia_match'] = False
                result['usia_note']  = (
                    f'Usia pelamar {usia} tahun melebihi batas maksimal '
                    f'{max_u} tahun yang disyaratkan loker'
                )
            else:
                result['usia_match'] = True
                usia_range = ''
                if min_u and max_u:
                    usia_range = f' (syarat: {min_u}–{max_u} tahun)'
                elif max_u:
                    usia_range = f' (syarat: maksimal {max_u} tahun)'
                elif min_u:
                    usia_range = f' (syarat: minimal {min_u} tahun)'
                result['usia_note'] = (
                    f'Usia pelamar {usia} tahun memenuhi syarat usia loker'
                    f'{usia_range}'
                )

        # 2. GENDER
        req_gender     = req.get('gender')
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

        # 3. PENDIDIKAN — cek gap terhadap minimum
        min_edu_req = req.get('min_edu', 0)

        if min_edu_req == 0:
            result['edu_gap']  = False
            result['edu_note'] = 'Loker tidak mencantumkan syarat pendidikan minimum'

        elif pelamar_edu_kategori is None:
            result['edu_gap']  = False
            result['edu_note'] = 'Data pendidikan pelamar tidak tersedia'

        else:
            pelamar_edu_level = _edu_level_from_str(pelamar_edu_kategori)
            edu_label_map = {
                10: 'S3', 8: 'S2', 6: 'S1/D4', 5: 'D3',
                4: 'D1/D2', 3: 'SMA/SMK', 2: 'SMP', 1: 'SD', 0: '-'
            }
            req_label     = edu_label_map.get(min_edu_req, str(min_edu_req))
            pelamar_label = pelamar_edu_kategori

            if pelamar_edu_level >= min_edu_req:
                result['edu_gap']  = False
                result['edu_note'] = (
                    f'Pendidikan pelamar ({pelamar_label}) memenuhi '
                    f'syarat minimum loker ({req_label})'
                )
            else:
                result['edu_gap']  = True
                result['edu_note'] = (
                    f'Pendidikan pelamar ({pelamar_label}) di bawah '
                    f'syarat minimum loker ({req_label})'
                )

        # 4. PENGALAMAN — cek gap terhadap minimum
        min_exp_req = req.get('min_exp_years')

        if min_exp_req is None:
            result['exp_gap']  = False
            result['exp_note'] = 'Loker tidak mencantumkan syarat pengalaman minimum'

        elif min_exp_req == 0:
            result['exp_gap']  = False
            result['exp_note'] = 'Loker terbuka untuk fresh graduate'

        else:
            exp_years = float(total_exp_years or 0)

            if exp_years >= min_exp_req:
                result['exp_gap']  = False
                result['exp_note'] = (
                    f'Total pengalaman pelamar '
                    f'({int(exp_years)} tahun) memenuhi '
                    f'syarat minimum loker ({min_exp_req} tahun)'
                )
            else:
                result['exp_gap']  = True
                result['exp_note'] = (
                    f'Total pengalaman pelamar '
                    f'({int(exp_years)} tahun) kurang dari '
                    f'syarat minimum loker ({min_exp_req} tahun)'
                )

        # 5. BAHASA — cek yang wajib tapi tidak ada di profil
        required_langs = req.get('required_langs', [])

        if not required_langs:
            result['lang_missing'] = []
            result['lang_note']    = 'Loker tidak mencantumkan syarat bahasa khusus'

        else:
            # Gabungkan skills + deskripsi diri untuk cek bahasa
            skills_text = ' '.join(
                str(s).lower() for s in (pelamar_skills_raw or [])
            )

            lang_keyword_map = {
                'Mandarin' : ['mandarin', 'tiongkok', 'chinese', '中文'],
                'Inggris'  : ['inggris', 'english'],
                'Indonesia': ['indonesia'],
            }

            missing = []
            for lang in required_langs:
                keywords = lang_keyword_map.get(lang, [lang.lower()])
                found = any(kw in skills_text for kw in keywords)
                if not found:
                    missing.append(lang)

            result['lang_missing'] = missing

            if not missing:
                result['lang_note'] = (
                    f'Pelamar memiliki kemampuan bahasa yang disyaratkan: '
                    f'{", ".join(required_langs)}'
                )
            else:
                result['lang_note'] = (
                    f'Bahasa yang disyaratkan loker namun tidak ditemukan '
                    f'di profil pelamar: {", ".join(missing)}'
                )

        return result
