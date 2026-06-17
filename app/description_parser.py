import re
from bs4 import BeautifulSoup

# KONSTANTA PATTERN — berdasarkan analisis 58 loker real
_NICE_TO_HAVE_MARKERS = [
    'menjadi nilai tambah',
    'nilai tambah',
    'diutamakan memiliki',      # "diutamakan memiliki X" = nice-to-have
    'lebih diutamakan',
    'dipertimbangkan',
    'menjadi keuntungan',
    'akan menjadi',
    'fresh graduate dipertimbangkan',
    'fresh graduate dipersilakan',
]

# Mapping keyword bahasa → nama standar
_LANG_KEYWORDS = {
    'mandarin' : 'Mandarin',
    'tiongkok' : 'Mandarin',
    'chinese'  : 'Mandarin',
    'inggris'  : 'Inggris',
    'english'  : 'Inggris',
    'indonesia': 'Indonesia',
}

# Markers bahasa WAJIB (bukan nilai tambah)
_LANG_REQUIRED_MARKERS = [
    'wajib', 'fasih', 'mahir', 'mampu', 'aktif',
    'menggunakan', 'berbahasa', 'bilingual',
]

# Edu level mapping (sama dengan scoring.py untuk konsistensi)
_EDU_LEVEL = {
    's3': 10, 'doktor': 10,
    's2': 8,  'magister': 8,
    's1': 6,  'sarjana': 6,
    'd4': 6,
    'd3': 5,  'diploma': 5,
    'd2': 4,
    'd1': 4,
    'smk': 3, 'sma': 3,
    'smp': 2,
    'sd' : 1,
}

# HELPER
def _clean_html(html: str) -> str:
    if not html:
        return ''
    return BeautifulSoup(str(html), 'html.parser').get_text(separator='\n')


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text).lower()).strip()


def _is_nice_to_have(sentence: str) -> bool:
    s = _normalize(sentence)
    return any(marker in s for marker in _NICE_TO_HAVE_MARKERS)


def _extract_sentences(plain_text: str) -> list:
    # Ganti delimiter umum dengan newline
    text = re.sub(r'[;\n]', '\n', plain_text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines


# EXTRACTOR: PENDIDIKAN MINIMUM
def _extract_min_edu(sentences: list) -> int:
    edu_found = 0

    for sent in sentences:
        s = _normalize(sent)

        # Skip nice-to-have
        if _is_nice_to_have(sent):
            continue

        # Cari trigger kata pendidikan
        triggers = ['minimal', 'minimum', 'lulusan', 'pendidikan', 'ijazah']
        if not any(t in s for t in triggers):
            continue

        # Scan semua edu keyword, ambil yang tertinggi disebutkan
        levels_in_sent = []
        for key, lvl in _EDU_LEVEL.items():
            if key in s:
                levels_in_sent.append(lvl)

        if levels_in_sent:
            # Ambil minimum dari yang disebutkan = syarat minimum
            candidate = min(levels_in_sent)
            # Update edu_found hanya jika lebih tinggi dari yang sudah ada
            if candidate > edu_found:
                edu_found = candidate

    return edu_found

# EXTRACTOR: GENDER
def _extract_gender(sentences: list) -> str | None:
    gender_map = {
        'perempuan': 'Perempuan',
        'wanita'   : 'Perempuan',
        'pria'     : 'Laki-laki',
        'laki-laki': 'Laki-laki',
        'laki laki': 'Laki-laki',
        'putra'    : 'Laki-laki',
        'putri'    : 'Perempuan',
    }

    for sent in sentences:
        s = _normalize(sent)
        for keyword, gender_value in gender_map.items():
            if keyword in s:
                return gender_value

    return None


# EXTRACTOR: USIA
def _extract_usia(sentences: list) -> tuple:
    min_usia = None
    max_usia = None

    for sent in sentences:
        s = _normalize(sent)

        # Pattern: "usia X–Y" atau "usia X-Y" atau "berusia X-Y"
        range_match = re.search(
            r'usia\s+(\d+)\s*[–\-]\s*(\d+)',
            s
        )
        if range_match:
            min_usia = int(range_match.group(1))
            max_usia = int(range_match.group(2))
            break

        # Pattern: "maksimal X tahun" atau "max X tahun"
        max_match = re.search(
            r'(?:maksimal|maximum|max|maks)\s+(\d+)\s*tahun',
            s
        )
        if max_match:
            max_usia = int(max_match.group(1))

        # Pattern: "minimal X tahun" dalam konteks usia
        if 'usia' in s or 'umur' in s:
            min_match = re.search(
                r'(?:minimal|minimum|min)\s+(\d+)\s*tahun',
                s
            )
            if min_match:
                min_usia = int(min_match.group(1))

    return min_usia, max_usia


# EXTRACTOR: PENGALAMAN MINIMUM
def _extract_min_exp(sentences: list) -> int | None:
    for sent in sentences:
        s = _normalize(sent)

        # Fresh graduate = 0 tahun pengalaman wajib
        if 'fresh graduate' in s and 'tidak wajib' in s:
            return 0
        if 'fresh graduate dipersilakan' in s:
            return 0

        # Skip nice-to-have
        if _is_nice_to_have(sent):
            continue

        # Harus ada kata kunci pengalaman
        if 'pengalaman' not in s and 'experience' not in s:
            continue

        # Skip kalimat yang angka utamanya adalah range usia, bukan tahun exp
        usia_range_in_sent = re.search(
            r'(?:usia|berusia|umur)\s+(\d+)\s*[–\-]\s*(\d+)\s*tahun', s
        )
        if usia_range_in_sent:
            # Ada range usia di kalimat yang sama → jangan parse exp dari sini
            continue

        # Pattern range: "2–5 tahun" atau "2-5 tahun"
        range_match = re.search(
            r'(\d+)\s*[–\-]\s*(\d+)\s*tahun',
            s
        )
        if range_match:
            return int(range_match.group(1))  # ambil minimum range

        # Pattern tunggal: "minimal X tahun" atau "X tahun"
        single_match = re.search(
            r'(?:minimal|minimum|min|least)?\s*(\d+)\s*tahun',
            s
        )
        if single_match:
            return int(single_match.group(1))

    return None


# EXTRACTOR: BAHASA WAJIB
def _extract_required_langs(sentences: list) -> list:
    required = set()

    for sent in sentences:
        s = _normalize(sent)

        # Skip nice-to-have
        if _is_nice_to_have(sent):
            continue

        # Harus ada marker bahasa wajib
        has_lang_trigger = any(m in s for m in _LANG_REQUIRED_MARKERS)
        if not has_lang_trigger:
            continue

        # Scan semua bahasa
        for keyword, lang_name in _LANG_KEYWORDS.items():
            if keyword in s:
                required.add(lang_name)

    return sorted(list(required))


# EXTRACTOR: NICE TO HAVE
def _extract_nice_to_have(sentences: list) -> list:
    nice = []
    for sent in sentences:
        if _is_nice_to_have(sent) and len(sent.strip()) > 10:
            nice.append(sent.strip())
    return nice


# MAIN PARSER
class DescriptionParser:
    def parse(self, html: str) -> dict:
        # 1. Strip HTML → plain text
        plain = _clean_html(html)

        # 2. Pecah jadi kalimat/bullet
        sentences = _extract_sentences(plain)

        # 3. Extract tiap komponen
        min_edu       = _extract_min_edu(sentences)
        gender        = _extract_gender(sentences)
        min_usia, max_usia = _extract_usia(sentences)
        min_exp       = _extract_min_exp(sentences)
        required_langs = _extract_required_langs(sentences)
        nice_to_have  = _extract_nice_to_have(sentences)

        # 4. Clean text untuk embedding (sama seperti sebelumnya)
        clean_text = re.sub(r'\s+', ' ', plain).strip().lower()

        return {
            'hard_requirements': {
                'min_edu'        : min_edu,
                'gender'         : gender,
                'min_usia'       : min_usia,
                'max_usia'       : max_usia,
                'min_exp_years'  : min_exp,
                'required_langs' : required_langs,
            },
            'nice_to_have': nice_to_have,
            'clean_text'  : clean_text,
        }

    def parse_batch(self, html_list: list) -> list:
        # Parse banyak deskripsi sekaligus.
        return [self.parse(html) for html in html_list]
