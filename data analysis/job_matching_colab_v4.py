# ============================================================
# JOB MATCHING — ABLATION STUDY NOTEBOOK (REVISI v4)
# Tugas Akhir: Sistem Rekomendasi Lowongan Career Day
# ============================================================
#
# CHANGELOG dari versi sebelumnya (job_matching_colab_fixed.py):
#
# 1. SKILL SCORE
#    - Threshold similarity: 0.35 -> 0.50
#    - Basis perhitungan DIBALIK: dari "berapa % skill pelamar yang
#      relevan ke loker" menjadi "berapa % skill yang DI-REQUIRE
#      LOKER (lowonganskills) yang ke-cover oleh pelamar" — lebih
#      pas buat content-based filtering dari sudut pandang loker.
#    - Matching nama skill: exact/fuzzy text match dulu (nama baku
#      dari masterskills vs nama bebas pelamar), baru fallback ke
#      semantic similarity (sentence embedding) kalau gak ketemu.
#      Urutan ini dipilih supaya hasil paling explainable duluan.
#    - Kalau loker belum punya entry di lowonganskills sama sekali,
#      fallback ke cara lama (cosine ke deskripsi loker keseluruhan)
#      supaya pelamar gak dirugikan gara-gara data loker yang belum
#      lengkap di-input.
#
# 2. EDUCATION SCORE
#    - Skala ordinal diseragamkan: SD=1, SMP=2, SMA/SMK=3, D1=4,
#      D2=5, D3=6, D4/S1=7, S2=8, S3=9 — PERSIS sama dengan skala di
#      kolom lowongans.minimal_pendidikan. Exact match, bukan
#      substring 'in' check lagi (karena data DB sudah konsisten).
#    - level_score dibuat FLEXIBLE relatif ke requirement loker
#      (lowongans.minimal_pendidikan) kalau loker punya requirement;
#      fallback ke skala absolut level/9 kalau loker gak syaratin
#      jenjang tertentu.
#    - Bobot kombinasi: 0.5*level_score + 0.5*jurusan_score
#      (sebelumnya 0.35/0.65).
#
# 3. EXPERIENCE SCORE
#    - Durasi dihitung dari TOTAL BULAN pengalaman (manfaatin kolom
#      pelamarpengalamen.bulanawal/bulanselesai yang baru), bukan
#      estimasi tahun doang seperti sebelumnya.
#    - Formula baru: 0.5*posisi_score + 0.5*durasi_score.
#      Komponen recency (exponential decay) DIHAPUS total — biar
#      makin sedikit faktor implisit yang susah dijelasin.
#    - durasi_score FLEXIBLE relatif ke
#      lowongans.minimal_pengalaman_bulan masing-masing loker (BUKAN
#      hardcode ">=5 tahun = skor 1.0" seperti sebelumnya). Loker
#      yang gak syaratin pengalaman (fresh-grad-friendly, requirement
#      0 bulan) otomatis ngasih durasi_score = 1.0 ke siapapun.
#
# 4. GENDER & USIA
#    - lowongans.preferensi_gender, usia_min, usia_max TIDAK masuk
#      weighted score numerik. Cuma jadi flag biner (match_gender,
#      match_usia) buat reasoning/explanation layer di aplikasi,
#      bukan buat ranking di notebook ini.
#
# 5. NEGATIVE SAMPLING
#    - Dari "100% hard negative" (loker kategori sama doang) jadi
#      CAMPURAN hard + random/easy negative. Proporsinya TIDAK
#      diasumsikan sepihak — diuji dulu lewat ablation tahap 1
#      (kandidat: 0%, 30%, 50%, 70%, 100% random), baru dipilih yang
#      paling masuk akal dari hasil datanya. 70% random : 30% hard
#      jadi salah satu kandidat (sesuai diskusi), bukan dipatok jadi
#      angka final tanpa pembanding.
#
# 6. TEKS LOWONGAN UNTUK EMBEDDING
#    - lowongan_text sekarang ikut nyantumin skills_required_text dan
#      jurusan_required_text (dari master table), bukan cuma deskripsi
#      HTML mentah — bikin semantic_score lebih akurat karena
#      requirement-nya udah terstruktur, bukan hasil tebak dari teks
#      bebas.
#
# Label: pilihan pelamar (implicit feedback)
# Limitasi: diakui sebagai proxy, bukan ground truth kecocokan
#
# ------------------------------------------------------------
# CATATAN PENTING — DEPENDENSI SQL VIEW:
# Script ini butuh kolom skills_detail (format "nama|tingkat" per
# skill, dipisah ';;') di v_pelamars_csv & v_dataset_matching_csv.
# Kalau view SQL kamu belum punya kolom ini, tambahin dulu di
# v_pelamar_skill_agg:
#
#   GROUP_CONCAT(DISTINCT CONCAT(ps.namaskill, '|', ps.keterangan)
#                ORDER BY ps.namaskill SEPARATOR ';;') AS skills_detail
#
# lalu tambahin `sk.skills_detail` ke SELECT list di v_pelamars_csv
# dan v_dataset_matching_csv.
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import math, re, warnings, difflib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
CURRENT_YEAR = datetime.now().year
print("Import selesai.")

# ========================================================================
# CELL 1 — LOAD DATA
# ========================================================================
from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = '/content/drive/MyDrive/Machine_Learning/TA'
df_lamaran  = pd.read_csv(f'{BASE_PATH}/v_dataset_matching.csv')
df_lowongan = pd.read_csv(f'{BASE_PATH}/v_lowongans.csv')
df_pelamar  = pd.read_csv(f'{BASE_PATH}/v_pelamars.csv')

REQUIRED_COLS = {
    'dataset_matching.csv': (df_lamaran, [
        'lamaran_id', 'idpelamar', 'idlowongan', 'idkategori_loker',
        'skills_detail', 'pendidikan_tertinggi', 'edu_level_int',
        'jurusan_tertinggi', 'posisi_text', 'total_bulan_exp',
        'minimal_pendidikan', 'minimal_pengalaman_bulan',
        'preferensi_gender', 'usia_min', 'usia_max',
        'skills_required_text', 'jurusan_required_text',
    ]),
    'all_lowongans.csv': (df_lowongan, [
        'idlowongan', 'idkategori_loker', 'kategori_loker',
        'skills_required_text', 'jurusan_required_text',
        'minimal_pendidikan', 'minimal_pengalaman_bulan',
        'preferensi_gender', 'usia_min', 'usia_max',
    ]),
    'all_pelamars.csv': (df_pelamar, [
        'idpelamar', 'skills_detail', 'pendidikan_tertinggi',
        'jurusan_tertinggi', 'edu_level_int', 'posisi_text',
        'total_bulan_exp', 'jeniskelamin', 'usia',
    ]),
}

print("Validasi kolom CSV (versi schema baru):")
for name, (df, cols) in REQUIRED_COLS.items():
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [ERROR] {name} — kolom tidak ada: {missing}")
        print(f"          Re-export CSV pakai view SQL versi terbaru (lihat catatan di header script).")
    else:
        print(f"  [OK] {name}")

print(f"\nRingkasan data:")
print(f"  Positive samples (lamaran) : {len(df_lamaran):,} record")
print(f"  Lowongan unik              : {df_lowongan['idlowongan'].nunique():,}")
print(f"  Pelamar unik               : {df_pelamar['idpelamar'].nunique():,}")
print(f"  Kategori loker             : {df_lowongan['idkategori_loker'].nunique():,} kategori")

# ========================================================================
# CELL 2 — EDA
# ========================================================================

EDU_LEVEL_MAP = {
    'SD': 1, 'SMP': 2, 'SMA/SMK': 3, 'D1': 4, 'D2': 5,
    'D3': 6, 'D4/S1': 7, 'S2': 8, 'S3': 9,
}

def edu_level_from_str(kategori_str):
    """
    Konversi string kategori pendidikan ke skor ordinal 1-9, SAMA
    PERSIS dengan skala lowongans.minimal_pendidikan. Exact match
    (bukan substring 'in' check lagi) karena nilai dari DB sekarang
    konsisten (SD, SMP, SMA/SMK, D1...S3).
    """
    s = str(kategori_str).strip().upper()
    if s in EDU_LEVEL_MAP:
        return EDU_LEVEL_MAP[s]
    aliases = {'SMA': 3, 'SMK': 3, 'S1': 7, 'D4': 7}  # jaga-jaga data lama
    return aliases.get(s, 0)

print("\n" + "=" * 60)
print("EDA — KARAKTERISTIK DATA")
print("=" * 60)

print("\n--- Test edu_level_from_str() ---")
for tc in list(EDU_LEVEL_MAP.keys()) + ['SMA', 'S1', '']:
    print(f"  '{tc}' -> {edu_level_from_str(tc)}")

print("\n--- Kelengkapan Profil Pelamar ---")
cols_check = {
    'Ada skill (skills_detail)' : ('skills_detail',        'text'),
    'Ada pendidikan'            : ('pendidikan_tertinggi',  'text'),
    'Ada jurusan'               : ('jurusan_tertinggi',     'text'),
    'Ada pengalaman (posisi)'   : ('posisi_text',           'text'),
    'Ada total bulan exp'       : ('total_bulan_exp',       'numeric'),
}
for label, (col, dtype) in cols_check.items():
    if col not in df_pelamar.columns:
        print(f"  {label:30} : kolom tidak ada"); continue
    pct = (df_pelamar[col].fillna(0).gt(0).mean() * 100 if dtype == 'numeric'
           else df_pelamar[col].notna().mean() * 100)
    print(f"  {label:30} : {pct:.1f}%")

print("\n--- Kelengkapan Requirement Loker (kolom baru) ---")
loker_cols_check = {
    'Ada skill requirement (lowonganskills)' : 'skills_required_text',
    'Ada jurusan requirement'                : 'jurusan_required_text',
    'Ada minimal_pendidikan'                 : 'minimal_pendidikan',
    'Ada minimal_pengalaman_bulan (>0)'      : 'minimal_pengalaman_bulan',
}
for label, col in loker_cols_check.items():
    if col not in df_lowongan.columns:
        print(f"  {label:45} : kolom tidak ada"); continue
    pct = (df_lowongan[col].fillna(0).gt(0).mean() * 100 if col == 'minimal_pengalaman_bulan'
           else df_lowongan[col].notna().mean() * 100)
    print(f"  {label:45} : {pct:.1f}%")

n_no_skill_req = df_lowongan['skills_required_text'].isna().sum() if 'skills_required_text' in df_lowongan.columns else 0
if n_no_skill_req > 0:
    print(f"\n  [INFO] {n_no_skill_req} loker belum punya skill requirement di "
          f"lowonganskills -> skill_score loker ini bakal pakai fallback "
          f"(cosine ke deskripsi keseluruhan).")

print("\n--- Distribusi edu_level_int (pelamar) ---")
if 'edu_level_int' in df_pelamar.columns:
    for k, v in df_pelamar['edu_level_int'].value_counts().sort_index().items():
        print(f"  Level {k} : {v} pelamar")

print("\n--- Distribusi Lamaran per Pelamar ---")
lpp = df_lamaran.groupby('idpelamar')['idlowongan'].count()
print(f"  Rata-rata : {lpp.mean():.1f} | Median : {lpp.median():.0f} | "
      f"Min-Max : {lpp.min()}-{lpp.max()}")

print("\n--- Distribusi Kategori Loker (top 10) ---")
if 'kategori_loker' in df_lowongan.columns:
    for k, v in df_lowongan['kategori_loker'].value_counts().head(10).items():
        print(f"  {str(k):40} : {v} loker")

# ========================================================================
# CELL 3 — PREPROCESSING TEKS
# ========================================================================

def clean_html(text):
    if not text or pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ")

def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', str(text).lower())
    return text.strip()

def normalize_skill_text(text):
    """Normalisasi lebih agresif dari normalize_text — buang tanda baca
    juga, dipakai khusus buat exact/fuzzy matching nama skill."""
    text = normalize_text(text)
    return re.sub(r'[^a-z0-9 ]', ' ', text).strip()

def build_pelamar_text(row):
    parts = []
    if pd.notna(row.get('skills_detail')) and str(row['skills_detail']).strip():
        nama_only = [it.split('|')[0].strip() for it in str(row['skills_detail']).split(';;') if it.strip()]
        parts.append(f"keahlian yang dimiliki: {', '.join(nama_only)}")
    if pd.notna(row.get('jurusan_tertinggi')) and str(row['jurusan_tertinggi']).strip():
        edu = row.get('pendidikan_tertinggi', '')
        parts.append(f"latar belakang pendidikan {edu} jurusan {row['jurusan_tertinggi']}")
    if pd.notna(row.get('posisi_text')) and str(row['posisi_text']).strip():
        parts.append(f"pengalaman kerja sebagai: {row['posisi_text']}")
    return normalize_text(" ".join(parts))

def build_lowongan_text(row):
    parts = []
    if pd.notna(row.get('namalowongan')) and str(row['namalowongan']).strip():
        parts.append(f"posisi yang dibutuhkan: {row['namalowongan']}")
    if pd.notna(row.get('kategori_loker')) and str(row['kategori_loker']).strip():
        parts.append(f"kategori pekerjaan: {row['kategori_loker']}")
    if pd.notna(row.get('skills_required_text')) and str(row['skills_required_text']).strip():
        parts.append(f"skill yang dibutuhkan: {row['skills_required_text']}")
    if pd.notna(row.get('jurusan_required_text')) and str(row['jurusan_required_text']).strip():
        parts.append(f"jurusan yang dibutuhkan: {row['jurusan_required_text']}")
    if pd.notna(row.get('deskripsi_loker')) and str(row['deskripsi_loker']).strip():
        bersih = clean_html(row['deskripsi_loker'])
        if bersih:
            parts.append(f"deskripsi pekerjaan: {bersih}")
    return normalize_text(" ".join(parts))

df_pelamar['pelamar_text']   = df_pelamar.apply(build_pelamar_text, axis=1)
df_lowongan['lowongan_text'] = df_lowongan.apply(build_lowongan_text, axis=1)

empty_p = (df_pelamar['pelamar_text'].str.strip()  == '').sum()
empty_l = (df_lowongan['lowongan_text'].str.strip() == '').sum()
print(f"Preprocessing selesai. Pelamar tanpa teks: {empty_p} | Lowongan tanpa teks: {empty_l}")
print(f"\nContoh pelamar_text :\n  {df_pelamar['pelamar_text'].iloc[0][:200]}")
print(f"\nContoh lowongan_text:\n  {df_lowongan['lowongan_text'].iloc[0][:200]}")

# ========================================================================
# CELL 4 — EMBEDDING
# ========================================================================
print("Loading model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded.")

pelamar_unique = df_pelamar.drop_duplicates('idpelamar')
print(f"Encoding {len(pelamar_unique)} profil pelamar...")
pelamar_vecs = model.encode(pelamar_unique['pelamar_text'].tolist(),
                             show_progress_bar=True, batch_size=64, convert_to_numpy=True)
pelamar_vec_dict = dict(zip(pelamar_unique['idpelamar'], pelamar_vecs))

print(f"Encoding {len(df_lowongan)} lowongan...")
lowongan_vecs = model.encode(df_lowongan['lowongan_text'].tolist(),
                              show_progress_bar=True, batch_size=64, convert_to_numpy=True)
lowongan_vec_dict = dict(zip(df_lowongan['idlowongan'], lowongan_vecs))
print(f"Embedding selesai. Dimensi: {pelamar_vecs.shape[1]}")

# ========================================================================
# CELL 5 — KONSTANTA, CACHE, & SKILL MATCHING
# ========================================================================

PROFICIENCY_WEIGHT = {'Kurang': 0.25, 'Cukup': 0.50, 'Baik': 0.75, 'Sangat Baik': 1.00}
SKILL_THRESHOLD = 0.50   # direvisi dari 0.35
FUZZY_THRESHOLD = 0.85   # ambang text-similarity fuzzy match, sebelum fallback semantic
                          # (tunable — turunin kalau match-nya kurang sensitif,
                          #  naikin kalau kebanyakan false-match gara2 typo)

_skill_vec_cache   = {}
_jurusan_vec_cache = {}
_exp_vec_cache     = {}

def get_cached_vec(text, cache_dict):
    if text not in cache_dict:
        cache_dict[text] = model.encode(text, convert_to_numpy=True)
    return cache_dict[text]

def skill_is_match(nama_pelamar, nama_required, threshold=SKILL_THRESHOLD):
    """
    Strategi matching skill (berhenti di tahap pertama yang match):
    1. Exact match (sesudah normalisasi teks)   -> paling explainable
    2. Fuzzy text similarity (difflib >= 0.85)  -> nutup typo/variasi tulis
    3. Semantic similarity (sentence embedding) -> fallback nutup sinonim
    """
    a, b = normalize_skill_text(nama_pelamar), normalize_skill_text(nama_required)
    if not a or not b:
        return False, 0.0, 'none'
    if a == b:
        return True, 1.0, 'exact'
    fuzzy = difflib.SequenceMatcher(None, a, b).ratio()
    if fuzzy >= FUZZY_THRESHOLD:
        return True, fuzzy, 'fuzzy'
    vec_a = get_cached_vec(f"memiliki keahlian {nama_pelamar}", _skill_vec_cache)
    vec_b = get_cached_vec(f"memiliki keahlian {nama_required}", _skill_vec_cache)
    sim = float(cos_sim([vec_a], [vec_b])[0][0])
    return sim >= threshold, sim, 'semantic'

print("Konstanta & skill matching siap.")
print("\n--- Test skill_is_match() ---")
for a, b in [("Microsoft Excel", "Excel"),
             ("Exel, Wor, PPT", "Microsoft Word"),
             ("Pemrograman Python", "Python Programming"),
             ("Las listrik", "Welding")]:
    ok, sim, method = skill_is_match(a, b)
    print(f"  '{a}' vs '{b}' -> match={ok} (sim={sim:.3f}, method={method})")

# ========================================================================
# CELL 6 — FUNGSI SCORING (S1-S4)
# ========================================================================

def semantic_score(pelamar_vec, lowongan_vec):
    """S1: Semantic Score — tidak berubah dari versi sebelumnya."""
    if pelamar_vec is None or lowongan_vec is None:
        return 0.0
    return round(float(cos_sim([pelamar_vec], [lowongan_vec])[0][0]), 4)

def _legacy_skill_score(skills_detail, lowongan_vec, threshold=SKILL_THRESHOLD):
    """Fallback: cara lama, basis skill pelamar vs cosine ke deskripsi
    loker keseluruhan. HANYA dipakai kalau loker belum punya skill
    requirement ter-tag di lowonganskills (data belum lengkap)."""
    if pd.isna(skills_detail) or not str(skills_detail).strip() or lowongan_vec is None:
        return 0.0
    total_w, matched_w = 0.0, 0.0
    for item in str(skills_detail).split(';;'):
        idx = item.rfind('|')
        if idx == -1:
            continue
        nama, level = item[:idx].strip(), item[idx + 1:].strip()
        w = PROFICIENCY_WEIGHT.get(level, 0.5)
        total_w += w
        vec = get_cached_vec(f"memiliki keahlian {nama}", _skill_vec_cache)
        sim = float(cos_sim([vec], [lowongan_vec])[0][0])
        if sim >= threshold:
            matched_w += w
    return round(matched_w / total_w, 4) if total_w > 0 else 0.0

def skill_score(skills_detail, skills_required_text, lowongan_vec, threshold=SKILL_THRESHOLD):
    """
    S2: Skill Score (REVISI, coverage-based).
    Basis: skill yang DI-REQUIRE LOKER (lowonganskills), bukan basis
    skill pelamar lagi.
        score = (Σ bobot proficiency utk requirement yang ke-cover)
                 / (jumlah total skill yang di-require loker)
    Kalau loker belum punya requirement ter-tag -> fallback cara lama.
    Return (score, method) — method buat tracking/QA.
    """
    if pd.isna(skills_required_text) or not str(skills_required_text).strip():
        return _legacy_skill_score(skills_detail, lowongan_vec, threshold), 'fallback_legacy'

    required = [s.strip() for s in str(skills_required_text).split(';') if s.strip()]
    if not required:
        return _legacy_skill_score(skills_detail, lowongan_vec, threshold), 'fallback_legacy'

    if pd.isna(skills_detail) or not str(skills_detail).strip():
        return 0.0, 'requirement_based'

    pelamar_skills = []
    for item in str(skills_detail).split(';;'):
        idx = item.rfind('|')
        if idx == -1:
            continue
        nama, level = item[:idx].strip(), item[idx + 1:].strip()
        pelamar_skills.append((nama, PROFICIENCY_WEIGHT.get(level, 0.5)))

    matched_total = 0.0
    for req in required:
        best_w = 0.0
        for nama, w in pelamar_skills:
            is_match, _, _ = skill_is_match(nama, req, threshold)
            if is_match:
                best_w = max(best_w, w)
        matched_total += best_w

    return round(matched_total / len(required), 4), 'requirement_based'

def edu_score(pendidikan_tertinggi, jurusan_tertinggi, lowongan_vec, minimal_pendidikan=None):
    """
    S3: Education Score (REVISI).
    - level_score: relatif ke minimal_pendidikan loker kalau ada
      (flexible per loker); fallback skala absolut level/9 kalau loker
      gak punya requirement spesifik.
    - jurusan_score: semantic similarity jurusan vs loker (sama seperti
      sebelumnya).
    - kombinasi: 0.5*level_score + 0.5*jurusan_score (sebelumnya 0.35/0.65)
    """
    level_int = edu_level_from_str(pendidikan_tertinggi) if pd.notna(pendidikan_tertinggi) else 0

    has_requirement = pd.notna(minimal_pendidikan) and float(minimal_pendidikan) > 0
    if has_requirement:
        level_score = min(level_int / float(minimal_pendidikan), 1.0) if level_int > 0 else 0.0
    else:
        level_score = level_int / 9.0 if level_int > 0 else 0.2

    if pd.notna(jurusan_tertinggi) and str(jurusan_tertinggi).strip() and lowongan_vec is not None:
        jurusan_vec = get_cached_vec(f"latar belakang pendidikan jurusan {jurusan_tertinggi}", _jurusan_vec_cache)
        jurusan_score = float(cos_sim([jurusan_vec], [lowongan_vec])[0][0])
    else:
        jurusan_score = 0.3

    return round(0.5 * level_score + 0.5 * jurusan_score, 4)

def exp_score(posisi_text, total_bulan_exp, lowongan_vec, minimal_pengalaman_bulan=0):
    """
    S4: Experience Score (REVISI).
    - posisi_score: relevansi semantic posisi TERBAIK yang pernah
      dijabat pelamar, vs loker.
    - durasi_score: total bulan pengalaman pelamar relatif ke
      minimal_pengalaman_bulan loker (flexible per loker; loker yang
      gak syaratin pengalaman -> durasi_score = 1.0 buat siapapun,
      termasuk fresh graduate).
    - kombinasi: 0.5*posisi_score + 0.5*durasi_score (recency DIHAPUS)
    """
    total_bulan_exp = float(total_bulan_exp) if pd.notna(total_bulan_exp) else 0.0
    min_req = float(minimal_pengalaman_bulan) if pd.notna(minimal_pengalaman_bulan) else 0.0

    durasi_score = 1.0 if min_req <= 0 else min(total_bulan_exp / min_req, 1.0)

    posisi_score = 0.0
    if pd.notna(posisi_text) and str(posisi_text).strip() and lowongan_vec is not None:
        sims = []
        for posisi in str(posisi_text).split(';'):
            posisi = posisi.strip()
            if not posisi:
                continue
            vec = get_cached_vec(f"pengalaman kerja sebagai {posisi}", _exp_vec_cache)
            sims.append(float(cos_sim([vec], [lowongan_vec])[0][0]))
        posisi_score = max(sims) if sims else 0.0

    return round(0.5 * posisi_score + 0.5 * durasi_score, 4)

print("Fungsi scoring siap (semantic, skill, edu, exp).")

# ========================================================================
# CELL 7 — MATCH FLAGS (reasoning, bukan score) & WRAPPER SCORING
# ========================================================================

def compute_match_flags(df):
    """match_gender & match_usia: TIDAK masuk weighted score, cuma flag
    biner buat reasoning/explanation layer (sesuai requirement)."""
    df = df.copy()

    def _gender_ok(r):
        pref = r.get('preferensi_gender')
        if pd.isna(pref) or pref in (None, 'Semua'):
            return 1
        return 1 if pref == r.get('jeniskelamin') else 0

    def _usia_ok(r):
        usia = r.get('usia')
        if pd.isna(usia):
            return 0
        umin = r.get('usia_min') or 0
        umax = r.get('usia_max') or 0
        ok_min = pd.isna(umin) or umin == 0 or usia >= umin
        ok_max = pd.isna(umax) or umax == 0 or usia <= umax
        return 1 if (ok_min and ok_max) else 0

    df['match_gender'] = df.apply(_gender_ok, axis=1)
    df['match_usia']   = df.apply(_usia_ok, axis=1)
    return df

def compute_all_scores(df, show_progress=True):
    s_semantic, s_skill, s_edu, s_exp, s_method = [], [], [], [], []
    iterator = tqdm(df.iterrows(), total=len(df), desc="Scoring") if show_progress else df.iterrows()
    for _, row in iterator:
        pv = pelamar_vec_dict.get(row.get('idpelamar'))
        lv = lowongan_vec_dict.get(row.get('idlowongan'))
        if pv is None or lv is None:
            s_semantic.append(0.0); s_skill.append(0.0); s_edu.append(0.0); s_exp.append(0.0)
            s_method.append('no_vector')
            continue
        s_semantic.append(semantic_score(pv, lv))
        sk, method = skill_score(row.get('skills_detail'), row.get('skills_required_text'), lv)
        s_skill.append(sk); s_method.append(method)
        s_edu.append(edu_score(row.get('pendidikan_tertinggi'), row.get('jurusan_tertinggi'),
                                lv, row.get('minimal_pendidikan')))
        s_exp.append(exp_score(row.get('posisi_text'), row.get('total_bulan_exp'),
                                lv, row.get('minimal_pengalaman_bulan')))
    out = df.copy()
    out['semantic_score']     = s_semantic
    out['skill_score']        = s_skill
    out['edu_score']          = s_edu
    out['exp_score']          = s_exp
    out['skill_score_method'] = s_method
    return out

print("Wrapper scoring siap.")

# ========================================================================
# CELL 8 — NEGATIVE SAMPLING (CAMPURAN HARD + RANDOM)
# ========================================================================

NEG_PER_POS = 2  # rasio total negative:positive tetap 1:2 seperti sebelumnya

LOKER_FEATURE_COLS = [
    'namalowongan', 'deskripsi_loker', 'kategori_loker',
    'minimal_pendidikan', 'minimal_pengalaman_bulan', 'preferensi_gender',
    'usia_min', 'usia_max', 'skills_required_text', 'jurusan_required_text',
    'lowongan_text',
]

def generate_negative_samples(df_lamaran, df_lowongan, df_pelamar,
                               random_ratio=0.7, neg_per_pos=NEG_PER_POS, seed=42):
    """
    Generate negative sample CAMPURAN:
    - random_ratio   -> proporsi 'easy' negative: loker diambil RANDOM
                         dari SELURUH pool loker (lintas kategori).
    - 1-random_ratio -> proporsi 'hard' negative: loker dari kategori
                         yang SAMA dengan yang pernah dilamar, yang
                         belum pernah dia lamar.
    random_ratio=1.0 -> full random. random_ratio=0.0 -> full hard
    (perilaku identik versi sebelumnya).
    """
    rng = np.random.default_rng(seed)

    applied_dict = df_lamaran.groupby('idpelamar')['idlowongan'].apply(set).to_dict()
    pelamar_kategori_dict = df_lamaran.groupby('idpelamar')['idkategori_loker'].apply(set).to_dict()
    kat_to_lowongan = df_lowongan.groupby('idkategori_loker')['idlowongan'].apply(list).to_dict()
    all_lowongan_ids = df_lowongan['idlowongan'].tolist()
    lowongan_kat_map = dict(zip(df_lowongan['idlowongan'], df_lowongan['idkategori_loker']))

    negative_rows = []

    for pelamar_id, applied_set in applied_dict.items():
        kategori_set = pelamar_kategori_dict.get(pelamar_id, set())
        n_pos    = len(applied_set)
        n_total  = n_pos * neg_per_pos
        n_random = int(round(n_total * random_ratio))
        n_hard   = n_total - n_random

        hard_candidates = [
            (lid, kat_id) for kat_id in kategori_set
            for lid in kat_to_lowongan.get(kat_id, [])
            if lid not in applied_set
        ]
        random_candidates = [
            (lid, lowongan_kat_map.get(lid)) for lid in all_lowongan_ids
            if lid not in applied_set
        ]

        chosen = []
        if n_hard > 0:
            if hard_candidates:
                idx = rng.choice(len(hard_candidates), size=n_hard,
                                  replace=len(hard_candidates) < n_hard)
                chosen += [hard_candidates[i] for i in idx]
            else:
                n_random += n_hard  # gak ada hard candidate -> alihin ke random

        if n_random > 0 and random_candidates:
            idx = rng.choice(len(random_candidates), size=n_random,
                              replace=len(random_candidates) < n_random)
            chosen += [random_candidates[i] for i in idx]

        if not chosen:
            continue

        pelamar_row = df_pelamar[df_pelamar['idpelamar'] == pelamar_id]
        if pelamar_row.empty:
            continue
        pelamar_data = pelamar_row.iloc[0].to_dict()

        for lid, kat_id in chosen:
            loker_row = df_lowongan[df_lowongan['idlowongan'] == lid]
            if loker_row.empty:
                continue
            loker_data = loker_row.iloc[0].to_dict()
            neg = pelamar_data.copy()
            neg.update({c: loker_data.get(c) for c in LOKER_FEATURE_COLS})
            neg['idlowongan']       = lid
            neg['idkategori_loker'] = kat_id
            neg['label']            = 0
            neg['lamaran_id']       = None
            negative_rows.append(neg)

    return pd.DataFrame(negative_rows)

print("Fungsi negative sampling siap.")

# ========================================================================
# CELL 9 — DATASET POSITIF (dihitung sekali, dipakai ulang di semua skenario)
# ========================================================================

df_positive = df_lamaran.copy()
df_positive['label'] = 1
df_positive = compute_match_flags(df_positive)

print("Menghitung skor untuk data POSITIF (dipakai ulang di semua skenario negative sampling)...")
df_positive_scored = compute_all_scores(df_positive)
print(f"Selesai. {len(df_positive_scored):,} baris positif sudah ada skornya.")

# ========================================================================
# CELL 10 — ABLATION TAHAP 1: RASIO NEGATIVE SAMPLING (random vs hard)
# ========================================================================
# Daripada pakai rasio tetap, beberapa kandidat diuji dulu pakai BOBOT
# NETRAL (0.25 semua) supaya perbandingan rasio gak ke-bias hasil tuning
# bobot (itu baru di Cell 12).

def ndcg_at_k(ranked_ids, relevant_ids, k=10):
    relevant_set = set(relevant_ids)
    dcg  = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked_ids[:k]) if r in relevant_set)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(ranked_ids, relevant_ids, k=10):
    relevant_set = set(relevant_ids)
    return sum(1 for r in ranked_ids[:k] if r in relevant_set) / k

def evaluate_weights(w1, w2, w3, w4, df_scores, k=10):
    tmp = df_scores.copy()
    tmp['final_score'] = (w1 * tmp['semantic_score'] + w2 * tmp['skill_score'] +
                           w3 * tmp['edu_score']      + w4 * tmp['exp_score'])
    ndcg_list, prec_list = [], []
    for pid, group in tmp.groupby('idpelamar'):
        relevant = group[group['label'] == 1]['idlowongan'].tolist()
        if not relevant:
            continue
        ranked = group.sort_values('final_score', ascending=False)['idlowongan'].tolist()
        ndcg_list.append(ndcg_at_k(ranked, relevant, k))
        prec_list.append(precision_at_k(ranked, relevant, k))
    return {'ndcg': round(np.mean(ndcg_list), 5), 'precision': round(np.mean(prec_list), 5),
            'n_pelamar': len(ndcg_list)}

K_EVAL = 10
RATIO_CANDIDATES = [0.0, 0.3, 0.5, 0.7, 1.0]  # proporsi RANDOM dalam negative sample

ratio_results   = []
negative_cache  = {}  # simpan df_negative_scored per rasio, dipakai lagi di Cell 11

for r in RATIO_CANDIDATES:
    print(f"\n--- Rasio random:hard = {r:.0%}:{1 - r:.0%} ---")
    df_neg = generate_negative_samples(df_lamaran, df_lowongan, df_pelamar, random_ratio=r)
    df_neg = compute_match_flags(df_neg)
    df_neg_scored = compute_all_scores(df_neg, show_progress=False)
    negative_cache[r] = df_neg_scored

    df_combo = pd.concat([df_positive_scored, df_neg_scored], ignore_index=True)
    metrics  = evaluate_weights(0.25, 0.25, 0.25, 0.25, df_combo, k=K_EVAL)

    deltas = {col: (df_combo[df_combo.label == 1][col].mean() - df_combo[df_combo.label == 0][col].mean())
              for col in ['semantic_score', 'skill_score', 'edu_score', 'exp_score']}

    ratio_results.append({
        'random_ratio': r, 'n_negative': len(df_neg_scored),
        'ndcg_equal_w': metrics['ndcg'], 'precision_equal_w': metrics['precision'],
        **{f'delta_{k}': round(v, 4) for k, v in deltas.items()},
    })

df_ratio_results = pd.DataFrame(ratio_results)
print("\n" + "=" * 70)
print("HASIL ABLATION RASIO NEGATIVE SAMPLING (bobot netral 0.25 semua)")
print("=" * 70)
print(df_ratio_results.to_string(index=False))

best_ratio_row    = df_ratio_results.sort_values('ndcg_equal_w', ascending=False).iloc[0]
BEST_RANDOM_RATIO = float(best_ratio_row['random_ratio'])
print(f"\n>> Rasio terpilih dari data: {BEST_RANDOM_RATIO:.0%} random : {1 - BEST_RANDOM_RATIO:.0%} hard "
      f"(NDCG@{K_EVAL}={best_ratio_row['ndcg_equal_w']:.4f})")
print("   Mau override manual (misal tetap pakai 70:30 sesuai diskusi awal)?")
print("   Tinggal set BEST_RANDOM_RATIO = 0.7 di baris setelah blok ini.")

# ========================================================================
# CELL 11 — DATASET FINAL (pakai rasio terpilih) & DISTRIBUSI SCORE
# ========================================================================

_closest_ratio_key = min(negative_cache.keys(), key=lambda x: abs(x - BEST_RANDOM_RATIO))
df_negative_final  = negative_cache[_closest_ratio_key]
df_all = pd.concat([df_positive_scored, df_negative_final], ignore_index=True)

print(f"Dataset final: {len(df_all):,} baris "
      f"({len(df_positive_scored):,} positif, {len(df_negative_final):,} negatif, "
      f"rasio 1:{len(df_negative_final)/max(len(df_positive_scored),1):.2f})")

print("\nStatistik skor per komponen (positif vs negatif):")
print(df_all[['label', 'semantic_score', 'skill_score', 'edu_score', 'exp_score']]
      .groupby('label').describe().round(3).to_string())

print("\n--- Metode skill_score yang terpakai ---")
print(df_all['skill_score_method'].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
score_cols = ['semantic_score', 'skill_score', 'edu_score', 'exp_score']
titles     = ['S1: Semantic Score', 'S2: Skill Score (revisi)',
              'S3: Education Score (revisi)', 'S4: Experience Score (revisi)']
for ax, col, title in zip(axes.flat, score_cols, titles):
    pos = df_all[df_all['label'] == 1][col].dropna()
    neg = df_all[df_all['label'] == 0][col].dropna()
    ax.hist(pos, bins=30, alpha=0.6, label='Positif', color='steelblue', density=True)
    ax.hist(neg, bins=30, alpha=0.6, label='Negatif', color='coral', density=True)
    ax.axvline(pos.mean(), color='steelblue', linestyle='--', label=f'Mean pos={pos.mean():.3f}')
    ax.axvline(neg.mean(), color='coral', linestyle='--', label=f'Mean neg={neg.mean():.3f}')
    ax.set_title(title, fontweight='bold'); ax.legend(fontsize=8)
plt.suptitle(f'Distribusi Score (rasio negative random:hard = '
             f'{BEST_RANDOM_RATIO:.0%}:{1 - BEST_RANDOM_RATIO:.0%})',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('score_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

save_path = f'{BASE_PATH}/scores_computed.csv'
df_all.to_csv(save_path, index=False)
print(f"\nTersimpan: {save_path}")

# ========================================================================
# CELL 12 — ABLATION TAHAP 2: BOBOT KOMPONEN (full grid, pakai rasio terpilih)
# ========================================================================

steps  = [round(x, 2) for x in np.arange(0.05, 0.71, 0.05)]
combos = []
for w1 in steps:
    for w2 in steps:
        for w3 in steps:
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if 0.05 <= w4 <= 0.70:
                combos.append((w1, w2, w3, w4))

print(f"Total kombinasi bobot: {len(combos):,}")
results = []
for w1, w2, w3, w4 in tqdm(combos, desc="Ablation bobot"):
    m = evaluate_weights(w1, w2, w3, w4, df_all, k=K_EVAL)
    results.append({'w_semantic': w1, 'w_skill': w2, 'w_edu': w3, 'w_exp': w4,
                     'ndcg': m['ndcg'], 'precision': m['precision']})

df_results = pd.DataFrame(results).sort_values('ndcg', ascending=False).reset_index(drop=True)
print("\nTop 10 kombinasi bobot:")
print(df_results.head(10).to_string(index=False))

best = df_results.iloc[0]
print("\n" + "=" * 60)
print("BOBOT OPTIMAL (FINAL)")
print("=" * 60)
print(f"  Rasio negative random:hard : {BEST_RANDOM_RATIO:.0%}:{1 - BEST_RANDOM_RATIO:.0%}")
print(f"  Semantic   (w1) : {best['w_semantic']:.2f}")
print(f"  Skill      (w2) : {best['w_skill']:.2f}")
print(f"  Education  (w3) : {best['w_edu']:.2f}")
print(f"  Experience (w4) : {best['w_exp']:.2f}")
print(f"  NDCG@{K_EVAL}        : {best['ndcg']:.4f}")
print(f"  Precision@{K_EVAL}   : {best['precision']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top_n = df_results.head(100)
sc = axes[0].scatter(top_n['w_semantic'], top_n['w_skill'], c=top_n['ndcg'],
                      cmap='viridis', s=80, alpha=0.8)
axes[0].scatter(best['w_semantic'], best['w_skill'], color='red', s=250,
                marker='*', zorder=6, label='Terbaik')
plt.colorbar(sc, ax=axes[0], label='NDCG@10')
axes[0].set_xlabel('Bobot Semantic'); axes[0].set_ylabel('Bobot Skill'); axes[0].legend()

komponen = ['Semantic', 'Skill', 'Education', 'Experience']
bobots   = [best['w_semantic'], best['w_skill'], best['w_edu'], best['w_exp']]
bars = axes[1].bar(komponen, bobots, color=['steelblue', 'coral', 'seagreen', 'darkorange'], alpha=0.85)
for bar, val in zip(bars, bobots):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.2f}', ha='center', fontweight='bold')
axes[1].axhline(0.25, color='gray', linestyle=':', label='Equal weight')
axes[1].set_title('Bobot Optimal per Komponen'); axes[1].legend()
plt.tight_layout()
plt.savefig('ablation_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n--- Sensitivity Analysis: Top 20 ---")
top20 = df_results.head(20)
print(top20[['w_semantic', 'w_skill', 'w_edu', 'w_exp', 'ndcg']].to_string(index=False))
for col in ['w_semantic', 'w_skill', 'w_edu', 'w_exp']:
    print(f"  std dev {col:12} : {top20[col].std():.4f}")

df_all.to_csv(f'{BASE_PATH}/scores_final.csv', index=False)
df_results.to_csv(f'{BASE_PATH}/ablation_weights.csv', index=False)
df_ratio_results.to_csv(f'{BASE_PATH}/ablation_negative_ratio.csv', index=False)
print(f"\nSemua hasil tersimpan di {BASE_PATH}/")
