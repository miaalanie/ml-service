# ==============================================================
# JOB MATCHING — ABLATION STUDY NOTEBOOK (REVISI v2)
# Tugas Akhir: Sistem Rekomendasi Lowongan Career Day
#
# Perbaikan dari v1 berdasarkan data real:
#   1. edu_score(): pakai str.contains() bukan dict exact match
#      → karena DB menyimpan "D4/S1", "SMA/SMK" bukan "S1"/"SMA"
#   2. EDA: tambah cek nilai unik pendidikan_tertinggi
#      → untuk verifikasi format dari data asli
#   3. exp_score(): clamp durasi agar tidak negatif
#      → kasus "Magang(2025-2025)" = 0 tahun, bukan error
#   4. skill_score(): sudah benar — namaskill boleh mengandung koma
#      karena delimiter utama adalah | bukan koma
# ==============================================================


# ============================================================ #
# CELL 1 — INSTALL & IMPORT                                    #
# ============================================================ #

# !pip install sentence-transformers scikit-learn pandas numpy \
#              matplotlib seaborn tqdm beautifulsoup4 -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math, re, json, warnings
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
CURRENT_YEAR = datetime.now().year

print("✓ Import selesai.")
print(f"  Tahun referensi: {CURRENT_YEAR}")


# ============================================================ #
# CELL 2 — LOAD DATA                                           #
# ============================================================ #

# from google.colab import files
# uploaded = files.upload()

df_lamaran  = pd.read_csv('dataset_matching.csv')
df_lowongan = pd.read_csv('all_lowongans.csv')
df_pelamar  = pd.read_csv('all_pelamars.csv')

def validate_columns(df, required_cols, df_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  [ERROR] {df_name} — kolom tidak ada: {missing}")
        print(f"          Pastikan pakai SQL export versi terbaru.")
    else:
        print(f"  [OK] {df_name}")

print("Validasi kolom CSV:")
validate_columns(df_lamaran,  ['skills_detail', 'idkategori_loker', 'pengalaman_detail',
                                'pendidikan_tertinggi', 'edu_level_int'], 'dataset_matching.csv')
validate_columns(df_lowongan, ['idkategori_loker', 'idlowongan'],         'all_lowongans.csv')
validate_columns(df_pelamar,  ['skills_detail', 'idpelamar',
                                'pendidikan_tertinggi', 'edu_level_int'], 'all_pelamars.csv')

print(f"\nRingkasan data:")
print(f"  Positive samples (lamaran) : {len(df_lamaran):,} record")
print(f"  Lowongan unik              : {df_lowongan['idlowongan'].nunique():,}")
print(f"  Pelamar unik               : {df_pelamar['idpelamar'].nunique():,}")
print(f"  Kategori loker             : {df_lowongan['idkategori_loker'].nunique():,} kategori")


# ============================================================ #
# CELL 3 — EDA                                                 #
# ============================================================ #

print("\n" + "="*60)
print("EDA — KARAKTERISTIK DATA")
print("="*60)

# 3a. Kelengkapan profil pelamar
print("\n--- 3a. Kelengkapan Profil Pelamar ---")
cols_check = {
    'Ada deskripsi diri' : ('deskripsidiri',        'text'),
    'Ada skill'          : ('skills_raw',            'text'),
    'Ada skill detail'   : ('skills_detail',         'text'),
    'Ada pendidikan'     : ('pendidikan_tertinggi',  'text'),
    'Ada jurusan'        : ('jurusan_tertinggi',     'text'),
    'Ada pengalaman'     : ('jumlah_pengalaman',     'numeric'),
}
for label, (col, dtype) in cols_check.items():
    if col not in df_pelamar.columns:
        print(f"  {label:30} : kolom tidak ada")
        continue
    if dtype == 'numeric':
        pct = df_pelamar[col].fillna(0).gt(0).mean() * 100
    else:
        pct = df_pelamar[col].notna().mean() * 100
    print(f"  {label:30} : {pct:.1f}%")

# 3b. PENTING: cek nilai unik pendidikan_tertinggi dari data real
# Ini untuk verifikasi format sebelum scoring
print("\n--- 3b. Nilai Unik Pendidikan Tertinggi (dari data real) ---")
if 'pendidikan_tertinggi' in df_pelamar.columns:
    uniq_edu = df_pelamar['pendidikan_tertinggi'].value_counts()
    for k, v in uniq_edu.items():
        score = _edu_level_from_str(str(k)) if False else '(lihat cell 7)'
        print(f"  '{k}' : {v} pelamar")
    print(f"\n  [NOTE] Format dari DB bisa 'D4/S1', 'SMA/SMK' dll.")
    print(f"  edu_score() pakai str.contains() — sudah handle format ini.")

# 3c. Cek edu_level_int dari SQL — harusnya sudah tidak 0 semua
print("\n--- 3c. Distribusi edu_level_int (dari SQL view) ---")
if 'edu_level_int' in df_pelamar.columns:
    lv_dist = df_pelamar['edu_level_int'].value_counts().sort_index()
    print(f"  Nilai 0 (tidak terdeteksi) : {(df_pelamar['edu_level_int']==0).sum()} pelamar")
    for k, v in lv_dist.items():
        print(f"  Level {k:2} : {v} pelamar")
    if (df_pelamar['edu_level_int'] == 0).all():
        print("\n  [WARNING] Semua edu_level_int masih 0!")
        print("  Pastikan SQL view sudah diupdate ke versi LIKE-based.")

# 3d. Lamaran per pelamar
print("\n--- 3d. Distribusi Lamaran per Pelamar ---")
lpp = df_lamaran.groupby('idpelamar')['idlowongan'].count()
print(f"  Rata-rata : {lpp.mean():.1f}")
print(f"  Median    : {lpp.median():.0f}")
print(f"  Min - Max : {lpp.min()} - {lpp.max()}")
print(f"  Melamar 1 loker  : {(lpp == 1).sum()} pelamar")
print(f"  Melamar >3 loker : {(lpp  > 3).sum()} pelamar")

# 3e. Distribusi kategori loker
print("\n--- 3e. Distribusi Kategori Loker ---")
if 'kategori_loker' in df_lowongan.columns:
    for k, v in df_lowongan['kategori_loker'].value_counts().items():
        print(f"  {str(k):40} : {v} loker")

# 3f. Cek format skills_detail
print("\n--- 3f. Verifikasi Format skills_detail ---")
sample_sk = df_pelamar['skills_detail'].dropna()
if len(sample_sk) > 0:
    print(f"  Contoh 1: {sample_sk.iloc[0][:100]}")
    if len(sample_sk) > 1:
        print(f"  Contoh 2: {sample_sk.iloc[1][:100]}")
    # Hitung berapa yang multi-skill (ada ;;)
    multi = sample_sk.str.contains(';;', na=False).sum()
    print(f"  Punya multi-skill (ada ;;) : {multi} pelamar")
    print(f"  Single skill saja          : {len(sample_sk)-multi} pelamar")
else:
    print("  [WARNING] Semua skills_detail kosong.")

# 3g. Cek format pengalaman_detail
print("\n--- 3g. Verifikasi Format pengalaman_detail ---")
if 'pengalaman_detail' in df_pelamar.columns:
    sample_exp = df_pelamar['pengalaman_detail'].dropna()
    if len(sample_exp) > 0:
        print(f"  Contoh: {sample_exp.iloc[0][:150]}")
        # Verifikasi pattern regex
        pattern = re.compile(r'.+\(\d{4}-(\d{4}|skrg)\)')
        valid = sample_exp.apply(
            lambda x: all(pattern.match(s.strip()) for s in str(x).split(';;'))
        ).sum()
        print(f"  Format valid (regex match): {valid}/{len(sample_exp)}")
    else:
        print("  Tidak ada pelamar dengan pengalaman kerja.")


# ============================================================ #
# CELL 4 — PREPROCESSING TEKS                                  #
# ============================================================ #

def clean_html(text):
    """Strip HTML tags dari deskripsi loker (deskripsi pakai rich text)."""
    if not text or pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ")

def normalize_text(text):
    """Lowercase + whitespace normalization."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_pelamar_text(row):
    """
    Representasi teks profil pelamar untuk embedding.
    Pakai section headers sebagai anchor konteks.

    Kolom: deskripsidiri, skills_raw, pendidikan_tertinggi,
           jurusan_tertinggi, posisi_all
    """
    parts = []
    if pd.notna(row.get('deskripsidiri')) and str(row['deskripsidiri']).strip():
        parts.append(f"profil diri: {row['deskripsidiri']}")
    if pd.notna(row.get('skills_raw')) and str(row['skills_raw']).strip():
        parts.append(f"keahlian yang dimiliki: {row['skills_raw']}")
    if pd.notna(row.get('jurusan_tertinggi')) and str(row['jurusan_tertinggi']).strip():
        edu = row.get('pendidikan_tertinggi', '')
        parts.append(f"latar belakang pendidikan {edu} jurusan {row['jurusan_tertinggi']}")
    if pd.notna(row.get('posisi_all')) and str(row['posisi_all']).strip():
        parts.append(f"pengalaman kerja sebagai: {row['posisi_all']}")
    return normalize_text(" ".join(parts))

def build_lowongan_text(row):
    """
    Representasi teks lowongan untuk embedding.
    Kolom: namalowongan, kategori_loker, deskripsi_loker (HTML)
    """
    parts = []
    if pd.notna(row.get('namalowongan')) and str(row['namalowongan']).strip():
        parts.append(f"posisi yang dibutuhkan: {row['namalowongan']}")
    if pd.notna(row.get('kategori_loker')) and str(row['kategori_loker']).strip():
        parts.append(f"kategori pekerjaan: {row['kategori_loker']}")
    if pd.notna(row.get('deskripsi_loker')) and str(row['deskripsi_loker']).strip():
        bersih = clean_html(row['deskripsi_loker'])
        if bersih:
            parts.append(f"deskripsi pekerjaan: {bersih}")
    return normalize_text(" ".join(parts))

df_pelamar['pelamar_text']   = df_pelamar.apply(build_pelamar_text,  axis=1)
df_lowongan['lowongan_text'] = df_lowongan.apply(build_lowongan_text, axis=1)

empty_p = (df_pelamar['pelamar_text'].str.strip()   == '').sum()
empty_l = (df_lowongan['lowongan_text'].str.strip()  == '').sum()
print(f"✓ Preprocessing selesai.")
print(f"  Pelamar tanpa teks  : {empty_p} (fallback score 0)")
print(f"  Lowongan tanpa teks : {empty_l}")
print(f"\n  Contoh pelamar_text :\n  {df_pelamar['pelamar_text'].iloc[0][:200]}")
print(f"\n  Contoh lowongan_text:\n  {df_lowongan['lowongan_text'].iloc[0][:200]}")


# ============================================================ #
# CELL 5 — EMBEDDING                                           #
# ============================================================ #
#
# Model: paraphrase-multilingual-MiniLM-L12-v2
# Alasan: multilingual (support Bahasa Indonesia), ringan (MiniLM),
# dilatih untuk semantic similarity, output 384 dim.
# Referensi: Reimers & Gurevych (2019) Sentence-BERT

print("Loading model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✓ Model loaded.")

pelamar_unique = df_pelamar.drop_duplicates('idpelamar')
print(f"\nEncoding {len(pelamar_unique)} profil pelamar...")
pelamar_vecs = model.encode(
    pelamar_unique['pelamar_text'].tolist(),
    show_progress_bar=True, batch_size=64, convert_to_numpy=True
)
pelamar_vec_dict = dict(zip(pelamar_unique['idpelamar'], pelamar_vecs))

print(f"\nEncoding {len(df_lowongan)} lowongan...")
lowongan_vecs = model.encode(
    df_lowongan['lowongan_text'].tolist(),
    show_progress_bar=True, batch_size=64, convert_to_numpy=True
)
lowongan_vec_dict = dict(zip(df_lowongan['idlowongan'], lowongan_vecs))

print(f"\n✓ Embedding selesai. Dimensi: {pelamar_vecs.shape[1]}")


# ============================================================ #
# CELL 6 — GENERATE NEGATIVE SAMPLES                           #
# ============================================================ #
#
# Strategi: untuk setiap pelamar, ambil loker di kategori yang
# SAMA yang tidak dia lamar → label = 0
# Ratio 1:2 (positif:negatif) — He & Garcia (2009)

print("Generating negative samples...")

if 'idkategori_loker' not in df_lamaran.columns:
    raise ValueError(
        "Kolom 'idkategori_loker' tidak ada di dataset_matching.csv.\n"
        "Pastikan pakai SQL export versi terbaru (dengan LIKE-based view)."
    )

applied_dict = (
    df_lamaran.groupby('idpelamar')['idlowongan'].apply(set).to_dict()
)
pelamar_kategori_dict = (
    df_lamaran.groupby('idpelamar')['idkategori_loker'].apply(set).to_dict()
)
kat_to_lowongan = (
    df_lowongan.groupby('idkategori_loker')['idlowongan'].apply(list).to_dict()
)

negative_samples = []
NEGATIVE_RATIO   = 2
np.random.seed(42)

for pelamar_id, applied_set in tqdm(applied_dict.items(), desc="Generate negatives"):
    kategori_set = pelamar_kategori_dict.get(pelamar_id, set())
    if not kategori_set:
        continue

    candidate_negatives = []
    for kat_id in kategori_set:
        for lid in kat_to_lowongan.get(kat_id, []):
            if lid not in applied_set:
                candidate_negatives.append((lid, kat_id))

    if not candidate_negatives:
        continue

    n_positif  = len(applied_set)
    n_negatif  = min(len(candidate_negatives), n_positif * NEGATIVE_RATIO)
    idx_chosen = np.random.choice(len(candidate_negatives), size=n_negatif, replace=False)
    chosen     = [candidate_negatives[i] for i in idx_chosen]

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
        neg.update({
            'idlowongan'      : lid,
            'namalowongan'    : loker_data.get('namalowongan'),
            'deskripsi_loker' : loker_data.get('deskripsi_loker'),
            'kategori_loker'  : loker_data.get('kategori_loker'),
            'idkategori_loker': kat_id,
            'lowongan_text'   : loker_data.get('lowongan_text'),
            'label'           : 0,
            'lamaran_id'      : None,
            'statusditerima'  : None,
        })
        negative_samples.append(neg)

df_positive          = df_lamaran.copy()
df_positive['label'] = 1

# Pastikan kolom teks ada di positive
if 'pelamar_text' not in df_positive.columns:
    df_positive = df_positive.merge(
        df_pelamar[['idpelamar', 'pelamar_text']], on='idpelamar', how='left'
    )
if 'lowongan_text' not in df_positive.columns:
    df_positive = df_positive.merge(
        df_lowongan[['idlowongan', 'lowongan_text']], on='idlowongan', how='left'
    )

df_negative = pd.DataFrame(negative_samples)
df_all      = pd.concat([df_positive, df_negative], ignore_index=True)

print(f"\n✓ Negative sampling selesai.")
print(f"  Positive : {len(df_positive):,}")
print(f"  Negative : {len(df_negative):,}")
print(f"  Total    : {len(df_all):,}")
print(f"  Ratio    : 1:{len(df_negative)//max(len(df_positive),1)}")


# ============================================================ #
# CELL 7 — FUNGSI SCORING                                      #
# ============================================================ #

# --- Konstanta ---
PROFICIENCY_WEIGHT = {
    'Kurang'     : 0.25,
    'Cukup'      : 0.50,
    'Baik'       : 0.75,
    'Sangat Baik': 1.00
}

SKILL_THRESHOLD = 0.55
RECENCY_LAMBDA  = 0.15

# Cache encode
_skill_vec_cache   = {}
_jurusan_vec_cache = {}
_exp_vec_cache     = {}

def get_cached_vec(text, cache_dict):
    if text not in cache_dict:
        cache_dict[text] = model.encode(text, convert_to_numpy=True)
    return cache_dict[text]


# --- Helper edu level ---
def _edu_level_from_str(kategori_str):
    """
    FIX UTAMA: pakai str.contains() bukan dict exact match.

    Data real dari DB menyimpan "D4/S1", "SMA/SMK" dll.
    Exact match ke "S1" atau "SMA" tidak akan pernah cocok.

    Order pengecekan dari tertinggi ke terendah supaya tidak
    salah: "D4/S1" cek S3 dulu (F), S2 (F), S1 (T) → 6 ✓
    """
    s = str(kategori_str).upper().strip()
    if   'S3'  in s: return 10
    elif 'S2'  in s: return 8
    elif 'S1'  in s: return 6
    elif 'D4'  in s: return 6
    elif 'D3'  in s: return 5
    elif 'D2'  in s: return 4
    elif 'D1'  in s: return 4
    elif 'SMK' in s: return 3
    elif 'SMA' in s: return 3
    elif 'SMP' in s: return 2
    elif 'SD'  in s: return 1
    return 0


# --- S1: Semantic Score ---
def semantic_score(pelamar_vec, lowongan_vec):
    """Cosine similarity antara vektor holistik pelamar dan loker."""
    if pelamar_vec is None or lowongan_vec is None:
        return 0.0
    return round(float(cos_sim([pelamar_vec], [lowongan_vec])[0][0]), 4)


# --- S2: Skill Score ---
def skill_score(skills_detail, lowongan_vec, threshold=SKILL_THRESHOLD):
    """
    Formula: Σ(w_i × match_i) / Σ(w_i)
    w_i     = proficiency weight (Kurang=0.25 … Sangat Baik=1.0)
    match_i = 1 jika cos_sim(encode(skill), job_vec) >= threshold

    Input: "namaskill|level;;namaskill|level"
    CATATAN: namaskill boleh mengandung koma (delimiter = |)
    Contoh valid: "Exel, Wor, PPT, Sap R3|Sangat Baik"
    """
    if pd.isna(skills_detail) or not str(skills_detail).strip():
        return 0.0
    if lowongan_vec is None:
        return 0.0

    total_w   = 0.0
    matched_w = 0.0

    for item in str(skills_detail).split(';;'):
        # Split dari KANAN supaya nama yang mengandung | tidak rusak
        # (seharusnya tidak ada, tapi defensif)
        idx = item.rfind('|')
        if idx == -1:
            continue
        nama  = item[:idx].strip()
        level = item[idx+1:].strip()
        w     = PROFICIENCY_WEIGHT.get(level, 0.5)
        total_w += w

        skill_text = f"memiliki keahlian {nama}"
        skill_vec  = get_cached_vec(skill_text, _skill_vec_cache)
        similarity = float(cos_sim([skill_vec], [lowongan_vec])[0][0])
        if similarity >= threshold:
            matched_w += w

    return round(matched_w / total_w, 4) if total_w > 0 else 0.0


# --- S3: Education Score ---
def edu_score(pendidikan_tertinggi, jurusan_tertinggi, lowongan_vec):
    """
    Formula: 0.35 × level_score + 0.65 × jurusan_score

    FIX: level_score pakai _edu_level_from_str() yang handle
    format "D4/S1", "SMA/SMK" dll — bukan dict exact match.

    Pembagian 35/65: jurusan lebih diskriminatif dari jenjang.
    S1 Hukum vs S1 Informatika untuk loker IT → jenjang sama,
    relevansi sangat berbeda (Ritter, 2011).
    """
    # Dimensi 1: Level jenjang
    if pd.notna(pendidikan_tertinggi) and str(pendidikan_tertinggi).strip():
        level_int   = _edu_level_from_str(str(pendidikan_tertinggi))
        level_score = level_int / 10.0
    else:
        level_score = 0.2

    # Dimensi 2: Relevansi jurusan
    if (pd.notna(jurusan_tertinggi)
            and str(jurusan_tertinggi).strip()
            and lowongan_vec is not None):
        jurusan_text  = f"latar belakang pendidikan jurusan {jurusan_tertinggi}"
        jurusan_vec   = get_cached_vec(jurusan_text, _jurusan_vec_cache)
        jurusan_score = float(cos_sim([jurusan_vec], [lowongan_vec])[0][0])
    else:
        jurusan_score = 0.3

    return round(0.35 * level_score + 0.65 * jurusan_score, 4)


# --- S4: Experience Score ---
def exp_score(pengalaman_detail, lowongan_vec):
    """
    Formula per exp i:
      score_i = relevance_i × (0.6 + 0.2×dur_i + 0.2×rec_i)

    relevance_i = cos_sim(encode("pengalaman sebagai X"), job_vec)
    dur_i       = min(durasi/5, 1.0)   [cap 5 tahun, Schmidt 1986]
    rec_i       = exp(-λ × tahun_berlalu)

    Agregasi: max(score_i) — sufficient condition, fair buat career changer.

    FIX: GREATEST(durasi, 0) — handle "Magang(2025-2025)" = 0 tahun,
    bukan error negatif.

    Format input: "posisi(tahunAwal-tahunAkhir);;posisi(tahunAwal-skrg)"
    """
    if pd.isna(pengalaman_detail) or not str(pengalaman_detail).strip():
        return 0.1

    if lowongan_vec is None:
        return 0.1

    scores  = []
    pattern = re.compile(r'^(.+?)\((\d{4})-([\d]+|skrg)\)$')

    for item in str(pengalaman_detail).split(';;'):
        item = item.strip()
        m    = pattern.match(item)
        if not m:
            continue

        posisi, thn_awal_str, thn_akhir_str = m.groups()
        thn_awal  = int(thn_awal_str)
        thn_akhir = CURRENT_YEAR if thn_akhir_str == 'skrg' else int(thn_akhir_str)
        thn_akhir = max(thn_awal, min(thn_akhir, CURRENT_YEAR))  # clamp

        # Relevance
        posisi_text = f"pengalaman kerja sebagai {posisi.strip()}"
        posisi_vec  = get_cached_vec(posisi_text, _exp_vec_cache)
        relevance   = float(cos_sim([posisi_vec], [lowongan_vec])[0][0])

        # FIX: durasi max(0) supaya tidak negatif
        durasi = max(thn_akhir - thn_awal, 0)
        dur_w  = min(durasi / 5.0, 1.0)

        berlalu = max(CURRENT_YEAR - thn_akhir, 0)
        rec_w   = math.exp(-RECENCY_LAMBDA * berlalu)

        score_i = relevance * (0.6 + 0.2 * dur_w + 0.2 * rec_w)
        scores.append(score_i)

    return round(float(max(scores)), 4) if scores else 0.1


# Verifikasi edu level dari data real
print("✓ Semua fungsi scoring siap.")
print("\n--- Verifikasi _edu_level_from_str() dengan nilai dari data real ---")
test_cases = ['D4/S1', 'SMA/SMK', 'S1', 'SMA', 'SMK', 'D3', 'S2', 'S3', 'SMP', 'SD', '']
for tc in test_cases:
    print(f"  '{tc}' → level {_edu_level_from_str(tc)}")


# ============================================================ #
# CELL 8 — HITUNG SEMUA SCORE                                  #
# ============================================================ #

print("Menghitung semua score komponen...")
print(f"  Total baris : {len(df_all):,}\n")

s_semantic_list = []
s_skill_list    = []
s_edu_list      = []
s_exp_list      = []

pair_skill_cache = {}
pair_edu_cache   = {}
pair_exp_cache   = {}

for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Scoring"):
    pid = row.get('idpelamar')
    lid = row.get('idlowongan')
    pv  = pelamar_vec_dict.get(pid)
    lv  = lowongan_vec_dict.get(lid)

    if pv is None or lv is None:
        s_semantic_list.append(0.0)
        s_skill_list.append(0.0)
        s_edu_list.append(0.0)
        s_exp_list.append(0.1)
        continue

    s_semantic_list.append(semantic_score(pv, lv))

    sk_key = (pid, lid)
    if sk_key not in pair_skill_cache:
        pair_skill_cache[sk_key] = skill_score(row.get('skills_detail'), lv)
    s_skill_list.append(pair_skill_cache[sk_key])

    ed_key = (pid, lid)
    if ed_key not in pair_edu_cache:
        pair_edu_cache[ed_key] = edu_score(
            row.get('pendidikan_tertinggi'),
            row.get('jurusan_tertinggi'), lv
        )
    s_edu_list.append(pair_edu_cache[ed_key])

    ex_key = (pid, lid)
    if ex_key not in pair_exp_cache:
        pair_exp_cache[ex_key] = exp_score(row.get('pengalaman_detail'), lv)
    s_exp_list.append(pair_exp_cache[ex_key])

df_all['semantic_score'] = s_semantic_list
df_all['skill_score']    = s_skill_list
df_all['edu_score']      = s_edu_list
df_all['exp_score']      = s_exp_list

print("\n✓ Scoring selesai.")
print("\nStatistik per komponen (positif vs negatif):")
print(
    df_all[['label','semantic_score','skill_score','edu_score','exp_score']]
    .groupby('label').describe().round(3).to_string()
)

df_all.to_csv('scores_computed.csv', index=False)
print("\n  Tersimpan: scores_computed.csv")


# ============================================================ #
# CELL 9 — DISTRIBUSI SCORE                                    #
# ============================================================ #

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
score_cols = ['semantic_score', 'skill_score', 'edu_score', 'exp_score']
titles     = ['S1: Semantic Score', 'S2: Skill Score',
              'S3: Education Score', 'S4: Experience Score']

for ax, col, title in zip(axes.flat, score_cols, titles):
    pos = df_all[df_all['label'] == 1][col].dropna()
    neg = df_all[df_all['label'] == 0][col].dropna()
    ax.hist(pos, bins=30, alpha=0.6, label='Positif', color='steelblue', density=True)
    ax.hist(neg, bins=30, alpha=0.6, label='Negatif', color='coral',     density=True)
    ax.axvline(pos.mean(), color='steelblue', linestyle='--', linewidth=1.5,
               label=f'Mean pos={pos.mean():.3f}')
    ax.axvline(neg.mean(), color='coral',     linestyle='--', linewidth=1.5,
               label=f'Mean neg={neg.mean():.3f}')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Score'); ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.suptitle('Distribusi Score: Positif vs Negatif', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('score_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n--- Separasi per Komponen ---")
print(f"  {'Komponen':<20} {'Mean Pos':>10} {'Mean Neg':>10} {'Delta':>10}")
print(f"  {'-'*52}")
for col in score_cols:
    mp = df_all[df_all['label']==1][col].mean()
    mn = df_all[df_all['label']==0][col].mean()
    print(f"  {col:<20} {mp:>10.4f} {mn:>10.4f} {mp-mn:>+10.4f}")
print("\n  Delta positif besar → komponen diskriminatif.")


# ============================================================ #
# CELL 10 — ABLATION STUDY                                     #
# ============================================================ #

def ndcg_at_k(ranked_ids, relevant_ids, k=10):
    relevant_set = set(relevant_ids)
    dcg  = sum(1.0/math.log2(i+2) for i,r in enumerate(ranked_ids[:k]) if r in relevant_set)
    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(ranked_ids, relevant_ids, k=10):
    relevant_set = set(relevant_ids)
    return sum(1 for r in ranked_ids[:k] if r in relevant_set) / k

def evaluate_weights(w1, w2, w3, w4, df_scores, k=10):
    tmp = df_scores.copy()
    tmp['final_score'] = (
        w1 * tmp['semantic_score'] + w2 * tmp['skill_score'] +
        w3 * tmp['edu_score']      + w4 * tmp['exp_score']
    )
    ndcg_list, prec_list = [], []
    for pid, group in tmp.groupby('idpelamar'):
        relevant = group[group['label']==1]['idlowongan'].tolist()
        if not relevant:
            continue
        ranked = group.sort_values('final_score', ascending=False)['idlowongan'].tolist()
        ndcg_list.append(ndcg_at_k(ranked, relevant, k))
        prec_list.append(precision_at_k(ranked, relevant, k))
    return {
        'ndcg'      : round(np.mean(ndcg_list), 5),
        'precision' : round(np.mean(prec_list), 5),
        'n_pelamar' : len(ndcg_list)
    }

K_EVAL = 10
steps  = [round(x, 2) for x in np.arange(0.05, 0.71, 0.05)]
combos = []
for w1 in steps:
    for w2 in steps:
        for w3 in steps:
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if 0.05 <= w4 <= 0.70:
                combos.append((w1, w2, w3, w4))

print(f"Total kombinasi bobot: {len(combos):,}")
print("Menjalankan ablation study...")

results = []
for w1, w2, w3, w4 in tqdm(combos, desc="Ablation"):
    m = evaluate_weights(w1, w2, w3, w4, df_all, k=K_EVAL)
    results.append({'w_semantic':w1,'w_skill':w2,'w_edu':w3,'w_exp':w4,
                    'ndcg':m['ndcg'],'precision':m['precision']})

df_results = pd.DataFrame(results).sort_values('ndcg', ascending=False).reset_index(drop=True)
print(f"\n✓ Ablation selesai.")
print(f"\nTop 10 kombinasi bobot (NDCG@{K_EVAL}):")
print(df_results.head(10).to_string(index=False))


# ============================================================ #
# CELL 11 — ANALISIS HASIL ABLATION                            #
# ============================================================ #

best = df_results.iloc[0]

print("\n" + "="*60)
print("BOBOT OPTIMAL")
print("="*60)
print(f"  Semantic   (w1) : {best['w_semantic']:.2f}")
print(f"  Skill      (w2) : {best['w_skill']:.2f}")
print(f"  Education  (w3) : {best['w_edu']:.2f}")
print(f"  Experience (w4) : {best['w_exp']:.2f}")
print(f"  NDCG@{K_EVAL}        : {best['ndcg']:.4f}")
print(f"  Precision@{K_EVAL}   : {best['precision']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top_n = df_results.head(100)
sc = axes[0].scatter(top_n['w_semantic'], top_n['w_skill'],
                     c=top_n['ndcg'], cmap='viridis', s=80, alpha=0.8)
axes[0].scatter(best['w_semantic'], best['w_skill'],
                color='red', s=250, marker='*', zorder=6, label='Terbaik')
plt.colorbar(sc, ax=axes[0], label='NDCG@10')
axes[0].set_xlabel('Bobot Semantic (w1)'); axes[0].set_ylabel('Bobot Skill (w2)')
axes[0].set_title('Top 100 Kombinasi'); axes[0].legend()

komponen = ['Semantic\n(w1)','Skill\n(w2)','Education\n(w3)','Experience\n(w4)']
bobots   = [best['w_semantic'],best['w_skill'],best['w_edu'],best['w_exp']]
colors   = ['steelblue','coral','seagreen','darkorange']
bars     = axes[1].bar(komponen, bobots, color=colors, alpha=0.85, edgecolor='white')
for bar, val in zip(bars, bobots):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, max(bobots)+0.12)
axes[1].set_title('Bobot Optimal per Komponen'); axes[1].set_ylabel('Bobot')
axes[1].axhline(0.25, color='gray', linestyle=':', label='Equal weight')
axes[1].legend()

plt.tight_layout()
plt.savefig('ablation_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n--- Sensitivity Analysis: Top 20 ---")
top20 = df_results.head(20)
print(top20[['w_semantic','w_skill','w_edu','w_exp','ndcg']].to_string(index=False))
print(f"\n  Std dev bobot di top-20:")
for col in ['w_semantic','w_skill','w_edu','w_exp']:
    print(f"    {col:15} : {top20[col].std():.4f}")
print("\n  Std dev rendah → bobot stabil dan reliable.")


# ============================================================ #
# CELL 12 — SIMPAN HASIL                                       #
# ============================================================ #

df_results.to_csv('ablation_results.csv', index=False)

best_weights = {
    'w_semantic'      : float(best['w_semantic']),
    'w_skill'         : float(best['w_skill']),
    'w_edu'           : float(best['w_edu']),
    'w_exp'           : float(best['w_exp']),
    'ndcg_at_k'       : float(best['ndcg']),
    'precision_at_k'  : float(best['precision']),
    'k'               : K_EVAL,
    'model'           : 'paraphrase-multilingual-MiniLM-L12-v2',
    'skill_threshold' : SKILL_THRESHOLD,
    'recency_lambda'  : RECENCY_LAMBDA,
    'catatan_fix'     : (
        'edu_score() pakai str.contains() bukan dict exact match '
        'karena DB menyimpan format "D4/S1", "SMA/SMK". '
        'exp_score() clamp durasi ke max(0) untuk handle "Magang(2025-2025)". '
        'skill_score() pakai rfind(|) untuk handle namaskill mengandung koma.'
    ),
    'catatan_limitasi': (
        'Label dari implicit feedback (pilihan melamar) — bukan ground truth kecocokan. '
        'Mengandung selection bias dari konteks career day. '
        'Lihat Hu et al. (2008) untuk justifikasi pendekatan ini.'
    )
}

with open('optimal_weights.json', 'w', encoding='utf-8') as f:
    json.dump(best_weights, f, indent=2, ensure_ascii=False)

print("✓ Semua file tersimpan:")
print("  scores_computed.csv     → dataset + semua score")
print("  ablation_results.csv    → semua kombinasi + NDCG/Precision")
print("  ablation_results.png    → visualisasi scatter + bar")
print("  score_distributions.png → distribusi score pos vs neg")
print("  optimal_weights.json    → bobot optimal final")

print("\n" + "="*60)
print("SELESAI")
print("="*60)
print(f"  Bobot: w1={best['w_semantic']:.2f} | w2={best['w_skill']:.2f} "
      f"| w3={best['w_edu']:.2f} | w4={best['w_exp']:.2f}")
print(f"  NDCG@{K_EVAL}: {best['ndcg']:.4f}")
print(f"  Data : {len(df_all):,} baris ({len(df_positive):,} pos + {len(df_negative):,} neg)")
