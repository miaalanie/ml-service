# ==============================================================
# JOB MATCHING — ABLATION STUDY NOTEBOOK
# Tugas Akhir: Sistem Rekomendasi Lowongan Career Day
#
# Tujuan:
#   1. EDA — pahami karakteristik data
#   2. Generate negative samples
#   3. Hitung komponen score (semantic, skill, edu, exp)
#   4. Ablation study → bobot optimal
#   5. Evaluasi dengan Precision@K dan NDCG@K
#
# Label: pilihan pelamar (implicit feedback)
# Limitasi: diakui sebagai proxy, bukan ground truth kecocokan
# Referensi: Hu et al. (2008) - Collaborative Filtering for
#            Implicit Feedback Datasets
# ==============================================================


# ============================================================ #
# CELL 1 — INSTALL & IMPORT                                    #
# ============================================================ #

# !pip install sentence-transformers scikit-learn pandas numpy matplotlib seaborn tqdm -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math, re, warnings
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
CURRENT_YEAR = datetime.now().year
print("Import selesai.")


# ============================================================ #
# CELL 2 — LOAD DATA                                           #
# ============================================================ #
#
# Upload 3 file CSV dari MySQL:
#   1. dataset_matching.csv  → dari v_dataset_matching
#   2. all_lowongans.csv     → semua loker
#   3. all_pelamars.csv      → semua pelamar lengkap
#
# SQL untuk export ada di bagian bawah file ini.

# from google.colab import files
# uploaded = files.upload()

# Untuk testing lokal:
df_lamaran  = pd.read_csv('dataset_matching.csv')
df_lowongan = pd.read_csv('all_lowongans.csv')
df_pelamar  = pd.read_csv('all_pelamars.csv')

print(f"Lamaran (positive samples) : {len(df_lamaran):,} record")
print(f"Lowongan unik              : {df_lowongan['idlowongan'].nunique():,}")
print(f"Pelamar unik               : {df_pelamar['idpelamar'].nunique():,}")


# ============================================================ #
# CELL 3 — EDA                                                 #
# ============================================================ #

print("\n" + "="*55)
print("EDA — KARAKTERISTIK DATA")
print("="*55)

# 3a. Kelengkapan profil pelamar
cols_check = {
    'Ada skill'          : 'skills_raw',
    'Ada pendidikan'     : 'pendidikan_tertinggi',
    'Ada jurusan'        : 'jurusan_tertinggi',
    'Ada pengalaman'     : 'jumlah_pengalaman',
}

print("\n--- Kelengkapan Profil Pelamar ---")
for label, col in cols_check.items():
    if col in df_pelamar.columns:
        if col == 'jumlah_pengalaman':
            pct = df_pelamar[col].fillna(0).gt(0).mean() * 100
        else:
            pct = df_pelamar[col].notna().mean() * 100
        print(f"  {label:25} : {pct:.1f}%")

# 3b. Lamaran per pelamar
lpp = df_lamaran.groupby('idpelamar')['idlowongan'].count()
print(f"\n--- Lamaran per Pelamar ---")
print(f"  Rata-rata : {lpp.mean():.1f}")
print(f"  Median    : {lpp.median():.0f}")
print(f"  Min-Max   : {lpp.min()} - {lpp.max()}")
print(f"  Melamar 1 loker   : {(lpp==1).sum()} pelamar")
print(f"  Melamar >3 loker  : {(lpp>3).sum()} pelamar")

# Insight penting
print(f"\n[INSIGHT] Pelamar yang melamar banyak loker sekaligus")
print(f"  menunjukkan selection bias lebih tinggi.")
print(f"  Ini bagian dari limitasi yang harus ditulis di penelitian.")

# 3c. Distribusi pendidikan
if 'pendidikan_tertinggi' in df_pelamar.columns:
    edu_dist = df_pelamar['pendidikan_tertinggi'].value_counts()
    print(f"\n--- Distribusi Pendidikan Tertinggi ---")
    for k, v in edu_dist.items():
        print(f"  {str(k):6} : {v:4} pelamar ({v/len(df_pelamar)*100:.1f}%)")

# 3d. Distribusi kategori loker
if 'kategori_loker' in df_lowongan.columns:
    print(f"\n--- Distribusi Kategori Loker ---")
    kat = df_lowongan['kategori_loker'].value_counts()
    for k, v in kat.items():
        print(f"  {str(k):35} : {v} loker")


# ============================================================ #
# CELL 4 — PREPROCESSING TEXT                                  #
# ============================================================ #

def clean_html(text):
    if not text or pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ")

def normalize(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_pelamar_text(row):
    """
    Representasi teks profil pelamar.

    Menggunakan section headers untuk memberikan konteks
    kepada transformer. Landasan: contextual embedding
    bekerja lebih baik dengan anchor konteks (Reimers, 2019).
    """
    parts = []
    if pd.notna(row.get('deskripsidiri')):
        parts.append(f"profil diri: {row['deskripsidiri']}")
    if pd.notna(row.get('skills_raw')):
        parts.append(f"keahlian yang dimiliki: {row['skills_raw']}")
    if pd.notna(row.get('jurusan_tertinggi')):
        edu = row.get('pendidikan_tertinggi', '')
        parts.append(f"latar belakang pendidikan {edu} jurusan {row['jurusan_tertinggi']}")
    if pd.notna(row.get('posisi_all')):
        parts.append(f"pengalaman kerja sebagai: {row['posisi_all']}")
    return normalize(" ".join(parts))

def build_lowongan_text(row):
    """
    Representasi teks lowongan.
    """
    parts = []
    if pd.notna(row.get('namalowongan')):
        parts.append(f"posisi yang dibutuhkan: {row['namalowongan']}")
    if pd.notna(row.get('kategori_loker')):
        parts.append(f"kategori pekerjaan: {row['kategori_loker']}")
    if pd.notna(row.get('deskripsi_loker')):
        parts.append(f"deskripsi pekerjaan: {clean_html(row['deskripsi_loker'])}")
    return normalize(" ".join(parts))

# Terapkan
df_pelamar['pelamar_text']  = df_pelamar.apply(build_pelamar_text, axis=1)
df_lowongan['lowongan_text'] = df_lowongan.apply(build_lowongan_text, axis=1)

print("Preprocessing selesai.")
print(f"\nContoh pelamar_text:\n{df_pelamar['pelamar_text'].iloc[0][:200]}")
print(f"\nContoh lowongan_text:\n{df_lowongan['lowongan_text'].iloc[0][:200]}")


# ============================================================ #
# CELL 5 — EMBEDDING                                           #
# ============================================================ #
#
# Model: paraphrase-multilingual-MiniLM-L12-v2
#
# Alasan pemilihan:
# 1. Multilingual → support Bahasa Indonesia
# 2. Dilatih untuk semantic similarity (paraphrase task)
# 3. Ringan (MiniLM) → feasible di Colab free
# 4. Output 384 dim → cukup ekspresif
#
# Referensi: Reimers & Gurevych (2019)
#            "Sentence-BERT: Sentence Embeddings using
#             Siamese BERT-Networks"

print("Loading model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded.")

# Encode semua teks unik — jangan encode duplikat
print(f"\nEncoding {len(df_pelamar)} pelamar...")
pelamar_vecs = model.encode(
    df_pelamar['pelamar_text'].tolist(),
    show_progress_bar=True,
    batch_size=64
)
# Dict: idpelamar → vector
pelamar_vec_dict = dict(zip(df_pelamar['idpelamar'], pelamar_vecs))

print(f"\nEncoding {len(df_lowongan)} lowongan...")
lowongan_vecs = model.encode(
    df_lowongan['lowongan_text'].tolist(),
    show_progress_bar=True,
    batch_size=64
)
# Dict: idlowongan → vector
lowongan_vec_dict = dict(zip(df_lowongan['idlowongan'], lowongan_vecs))

print("Embedding selesai.")


# ============================================================ #
# CELL 6 — GENERATE NEGATIVE SAMPLES                           #
# ============================================================ #
#
# Strategi: untuk setiap pelamar, ambil loker di kategori
# yang SAMA yang tidak dia lamar → label = 0
#
# Kenapa kategori yang sama?
# Negative sample yang baik = loker yang mungkin dilihat pelamar
# tapi tidak dipilih. Bukan loker dari kategori yang
# sama sekali berbeda (itu terlalu mudah dibedakan).
#
# Ratio 1:2 (positif:negatif)
# Landasan: class imbalance literature — ratio 1:2 atau 1:3
# memberikan hasil lebih stabil daripada 1:1 atau 1:10
# (He & Garcia, 2009 — Learning from Imbalanced Data)

print("Generating negative samples...")

# Mapping: idpelamar → set idlowongan yang sudah dilamar
applied_dict = (
    df_lamaran.groupby('idpelamar')['idlowongan']
    .apply(set)
    .to_dict()
)

# Mapping: idkategori → list idlowongan
kat_to_lowongan = (
    df_lowongan.groupby('idkategori_loker')['idlowongan']
    .apply(list)
    .to_dict()
)

negative_samples = []
NEGATIVE_RATIO = 2  # 2 negatif per 1 positif

for pelamar_id, applied_set in applied_dict.items():

    # Ambil kategori loker yang pernah dilamar pelamar ini
    applied_rows = df_lamaran[df_lamaran['idpelamar'] == pelamar_id]
    kategori_set = set(applied_rows['idkategori_loker'].tolist()) \
        if 'idkategori_loker' in applied_rows.columns \
        else set()

    if not kategori_set:
        continue

    # Cari loker di kategori yang sama yang TIDAK dilamar
    candidate_negatives = []
    for kat_id in kategori_set:
        loker_di_kat = kat_to_lowongan.get(kat_id, [])
        for lid in loker_di_kat:
            if lid not in applied_set:
                candidate_negatives.append(lid)

    # Ambil secara random, max = NEGATIVE_RATIO × jumlah positif
    n_positif  = len(applied_set)
    n_negatif  = min(len(candidate_negatives), n_positif * NEGATIVE_RATIO)

    if n_negatif == 0:
        continue

    chosen_negatives = np.random.choice(
        candidate_negatives, size=n_negatif, replace=False
    )

    # Ambil data pelamar
    pelamar_row = df_pelamar[df_pelamar['idpelamar'] == pelamar_id]
    if pelamar_row.empty:
        continue

    for lid in chosen_negatives:
        loker_row = df_lowongan[df_lowongan['idlowongan'] == lid]
        if loker_row.empty:
            continue

        neg = pelamar_row.iloc[0].to_dict()
        neg.update({
            'idlowongan'    : lid,
            'namalowongan'  : loker_row.iloc[0].get('namalowongan'),
            'deskripsi_loker': loker_row.iloc[0].get('deskripsi_loker'),
            'kategori_loker': loker_row.iloc[0].get('kategori_loker'),
            'idkategori_loker': loker_row.iloc[0].get('idkategori_loker'),
            'lowongan_text' : loker_row.iloc[0].get('lowongan_text'),
            'label'         : 0
        })
        negative_samples.append(neg)

df_negative = pd.DataFrame(negative_samples)
df_positive = df_lamaran.copy()
df_positive['label'] = 1

# Gabung positive + negative
df_all = pd.concat([df_positive, df_negative], ignore_index=True)

print(f"Positive samples : {len(df_positive):,}")
print(f"Negative samples : {len(df_negative):,}")
print(f"Total dataset    : {len(df_all):,}")
print(f"Ratio pos:neg    : 1:{len(df_negative)//max(len(df_positive),1)}")


# ============================================================ #
# CELL 7 — FUNGSI SCORING                                      #
# ============================================================ #

# --- Konstanta ---

PROFICIENCY_WEIGHT = {
    'Kurang': 0.25, 'Cukup': 0.50,
    'Baik': 0.75,   'Sangat Baik': 1.00
}

EDU_LEVEL = {
    'SD': 1,  'SMP': 2, 'SMA': 3, 'SMK': 3,
    'D1': 4,  'D2': 4,  'D3': 5,  'D4': 6,
    'S1': 6,  'S2': 8,  'S3': 10
}

# Threshold cosine untuk skill match
# Nilai ini akan divalidasi via eksperimen manual (cell berikutnya)
SKILL_THRESHOLD = 0.55

# Lambda recency decay
# λ=0.15: exp 5thn lalu = 47%, exp 10thn lalu = 22%
# Referensi: Schmidt et al. (1986) - experience-performance relationship
RECENCY_LAMBDA = 0.15


# --- Skill Score ---

def skill_score(skills_detail, lowongan_vec, threshold=SKILL_THRESHOLD):
    """
    Formula:
      skill_score = Σ(w_i × match_i) / Σ(w_i)

    w_i     = proficiency weight skill ke-i
    match_i = 1 jika cos_sim(skill_vec, job_vec) >= threshold

    Landasan weighted:
    Proficiency adalah ordinal signal valid di data.
    Pelamar dengan skill "Sangat Baik" lebih kompeten
    dari "Kurang" meski skill name sama.
    Mengabaikan ini = informasi terbuang.

    Landasan cosine untuk match:
    Menghindari exact string match yang gagal untuk
    sinonim ("MS Excel" vs "Microsoft Excel", "ML" vs
    "Machine Learning"). Pre-trained model sudah
    encode relasi ini dari corpus besar.
    """
    if pd.isna(skills_detail) or not skills_detail:
        return 0.0

    total_w   = 0.0
    matched_w = 0.0

    for item in str(skills_detail).split(';;'):
        parts = item.strip().split('|')
        if len(parts) != 2:
            continue
        nama, level = parts[0].strip(), parts[1].strip()
        w = PROFICIENCY_WEIGHT.get(level, 0.5)
        total_w += w

        # Encode dalam konteks kalimat untuk representasi lebih baik
        skill_text = f"memiliki keahlian {nama}"
        skill_vec  = model.encode(skill_text)
        similarity = float(cos_sim([skill_vec], [lowongan_vec])[0][0])

        if similarity >= threshold:
            matched_w += w

    return round(matched_w / total_w, 4) if total_w > 0 else 0.0


# --- Education Score ---

def edu_score(pendidikan_tertinggi, jurusan_tertinggi, lowongan_vec):
    """
    Formula:
      edu_score = 0.35 × level_score + 0.65 × jurusan_score

    level_score   = EDU_LEVEL[kategori] / 10
    jurusan_score = cos_sim(encode(jurusan), job_vec)

    Pembagian 35/65:
    Jurusan lebih diskriminatif dari jenjang.
    S1 Hukum vs S1 Informatika untuk loker IT =
    jenjang sama, relevansi sangat berbeda.
    Didukung: literatur rekrutmen teknis lebih
    menekankan relevansi bidang studi (Ritter, 2011).

    Hanya ambil pendidikan tertinggi:
    Pendidikan tertinggi merepresentasikan kompetensi
    akademik terkini. Jenjang bawah sudah 'included'.
    """
    # Dimensi 1: Level jenjang
    if pd.notna(pendidikan_tertinggi):
        level_int   = EDU_LEVEL.get(str(pendidikan_tertinggi).upper().strip(), 0)
        level_score = level_int / 10.0
    else:
        level_score = 0.2  # tidak ada data = rendah

    # Dimensi 2: Relevansi jurusan
    if pd.notna(jurusan_tertinggi) and str(jurusan_tertinggi).strip():
        jurusan_text  = f"latar belakang pendidikan jurusan {jurusan_tertinggi}"
        jurusan_vec   = model.encode(jurusan_text)
        jurusan_score = float(cos_sim([jurusan_vec], [lowongan_vec])[0][0])
    else:
        jurusan_score = 0.3  # tidak ada jurusan = netral rendah

    return round(0.35 * level_score + 0.65 * jurusan_score, 4)


# --- Experience Score ---

def exp_score(pengalaman_detail, lowongan_vec):
    """
    Formula per pengalaman i:
      score_i = relevance_i × (0.6 + 0.2×dur_i + 0.2×rec_i)

    relevance_i = cos_sim(encode(posisi_i), job_vec)
    dur_i       = min(durasi_tahun / 5, 1.0)
    rec_i       = e^(-λ × tahun_berlalu)

    Agregasi: max(score_i)

    Landasan max vs mean:
    Sufficient condition — satu pengalaman relevan
    sudah cukup untuk qualify. Mean tidak fair untuk
    career changer yang ganti bidang.

    Landasan duration cap 5 tahun:
    Schmidt et al. (1986): marginal return of experience
    plateaus untuk posisi entry-mid level.

    Landasan recency decay (λ=0.15):
    Skill obsolescence — teknologi dan metode kerja
    berubah. Pengalaman lama makin kurang relevan.

    Limitasi yang diakui:
    Field 'posisi' sering terlalu pendek (1-3 kata).
    Signal semantic dari posisi pendek lebih lemah
    dari deskripsi lengkap.
    """
    if pd.isna(pengalaman_detail) or not pengalaman_detail:
        return 0.1  # tidak ada pengalaman = sangat rendah

    scores = []
    pattern = re.compile(r'^(.+?)\((\d{4})-([\d]+|skrg)\)$')

    for item in str(pengalaman_detail).split(';;'):
        item = item.strip()
        m = pattern.match(item)
        if not m:
            continue

        posisi, thn_awal, thn_akhir = m.groups()
        thn_awal  = int(thn_awal)
        thn_akhir = CURRENT_YEAR if thn_akhir == 'skrg' else int(thn_akhir)

        # Relevance
        posisi_text = f"pengalaman kerja sebagai {posisi.strip()}"
        posisi_vec  = model.encode(posisi_text)
        relevance   = float(cos_sim([posisi_vec], [lowongan_vec])[0][0])

        # Duration weight
        durasi   = max(thn_akhir - thn_awal, 0)
        dur_w    = min(durasi / 5.0, 1.0)

        # Recency weight
        berlalu  = max(CURRENT_YEAR - thn_akhir, 0)
        rec_w    = math.exp(-RECENCY_LAMBDA * berlalu)

        score_i  = relevance * (0.6 + 0.2 * dur_w + 0.2 * rec_w)
        scores.append(score_i)

    return round(float(max(scores)), 4) if scores else 0.1


# --- Semantic Score ---

def semantic_score(pelamar_vec, lowongan_vec):
    """
    Cosine similarity antara representasi holistik
    pelamar dan lowongan.

    Mengukur kecocokan overall secara semantik,
    tidak per-komponen.

    Referensi: Reimers & Gurevych (2019) SBERT
    """
    return round(float(cos_sim([pelamar_vec], [lowongan_vec])[0][0]), 4)

print("Fungsi scoring siap.")


# ============================================================ #
# CELL 8 — HITUNG SEMUA SCORE                                  #
# ============================================================ #
#
# Ini bagian paling lama — perlu encode per skill dan per exp.
# Estimasi waktu: 20-60 menit tergantung ukuran data dan GPU.
#
# Tips: pakai Colab Pro atau aktifkan GPU runtime.

print("Menghitung semua score...")
print("Estimasi waktu: 20-60 menit\n")

s_semantic, s_skill, s_edu, s_exp = [], [], [], []

# Cache untuk hindari encode ulang
skill_cache = {}
exp_cache   = {}
edu_cache   = {}

for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
    pid = row.get('idpelamar')
    lid = row.get('idlowongan')

    pv = pelamar_vec_dict.get(pid)
    lv = lowongan_vec_dict.get(lid)

    # Fallback kalau vector tidak ada
    if pv is None or lv is None:
        s_semantic.append(0.0)
        s_skill.append(0.0)
        s_edu.append(0.0)
        s_exp.append(0.0)
        continue

    # Semantic — dari precomputed vector
    s_semantic.append(semantic_score(pv, lv))

    # Skill — cache per (pelamar, loker) pair
    sk_key = (pid, lid)
    if sk_key not in skill_cache:
        skill_cache[sk_key] = skill_score(
            row.get('skills_detail'), lv
        )
    s_skill.append(skill_cache[sk_key])

    # Edu — cache per (pelamar, loker) pair
    edu_key = (pid, lid)
    if edu_key not in edu_cache:
        edu_cache[edu_key] = edu_score(
            row.get('pendidikan_tertinggi'),
            row.get('jurusan_tertinggi'),
            lv
        )
    s_edu.append(edu_cache[edu_key])

    # Exp — cache per (pelamar, loker) pair
    exp_key = (pid, lid)
    if exp_key not in exp_cache:
        exp_cache[exp_key] = exp_score(
            row.get('pengalaman_detail'), lv
        )
    s_exp.append(exp_cache[exp_key])

df_all['semantic_score'] = s_semantic
df_all['skill_score']    = s_skill
df_all['edu_score']      = s_edu
df_all['exp_score']      = s_exp

print("\nSemua score selesai dihitung.")
print(df_all[['label','semantic_score','skill_score',
              'edu_score','exp_score']].describe().round(3))


# ============================================================ #
# CELL 9 — ANALISIS DISTRIBUSI SCORE                           #
# ============================================================ #
#
# Sebelum ablation, pahami dulu distribusi score.
# Ini penting untuk interpretasi hasil ablation.

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
score_cols = ['semantic_score', 'skill_score', 'edu_score', 'exp_score']
titles     = ['Semantic Score', 'Skill Score', 'Education Score', 'Experience Score']

for ax, col, title in zip(axes.flat, score_cols, titles):
    pos = df_all[df_all['label'] == 1][col]
    neg = df_all[df_all['label'] == 0][col]
    ax.hist(pos, bins=30, alpha=0.6, label='Positif (dilamar)', color='steelblue')
    ax.hist(neg, bins=30, alpha=0.6, label='Negatif', color='coral')
    ax.set_title(title)
    ax.set_xlabel('Score')
    ax.set_ylabel('Frekuensi')
    ax.legend()

plt.suptitle('Distribusi Score: Positif vs Negatif', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('score_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n[INSIGHT PENTING]")
print("Kalau distribusi positif dan negatif overlap banyak pada suatu score,")
print("komponen itu kurang diskriminatif → bobotnya mungkin lebih rendah.")
print("Kalau distribusi terpisah jelas → komponen itu sangat diskriminatif.")


# ============================================================ #
# CELL 10 — ABLATION STUDY                                     #
# ============================================================ #
#
# Tujuan: cari kombinasi bobot (w1,w2,w3,w4) yang menghasilkan
# ranking paling konsisten dengan pilihan historis pelamar.
#
# Metrik: NDCG@K (Normalized Discounted Cumulative Gain)
# Landasan pemilihan NDCG:
# - Mempertimbangkan posisi ranking, tidak hanya ada/tidak
# - Standard metric di information retrieval dan recsys
# - Referensi: Järvelin & Kekäläinen (2002)
#
# Constraint bobot:
# w1 + w2 + w3 + w4 = 1.0
# Setiap bobot: 0.05 sampai 0.70
# Step: 0.05

def ndcg_at_k(ranked_ids, relevant_ids, k=10):
    """
    NDCG@K = DCG@K / IDCG@K

    DCG@K  = Σ relevance_i / log2(i+2)  untuk i=0..k-1
    IDCG@K = DCG dari ranking ideal (semua relevan di atas)

    Skor 1.0 = semua loker relevan ada di top-K dengan urutan sempurna
    Skor 0.0 = tidak ada loker relevan di top-K
    """
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k]):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
    ideal_len = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(ranked_ids, relevant_ids, k=10):
    """
    Precision@K = |{top-K} ∩ {relevan}| / K
    """
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in ranked_ids[:k] if rid in relevant_set)
    return hits / k

def evaluate_weights(w1, w2, w3, w4, df_scores, k=10):
    """
    Untuk setiap pelamar:
      1. Hitung final score semua loker dengan bobot ini
      2. Rank loker berdasarkan final score
      3. Hitung NDCG@K dan Precision@K
      4. Rata-rata semua pelamar
    """
    df_scores = df_scores.copy()
    df_scores['final_score'] = (
        w1 * df_scores['semantic_score'] +
        w2 * df_scores['skill_score']    +
        w3 * df_scores['edu_score']      +
        w4 * df_scores['exp_score']
    )

    ndcg_list = []
    prec_list = []

    for pid, group in df_scores.groupby('idpelamar'):
        # Loker yang dilamar = relevan
        relevant = group[group['label'] == 1]['idlowongan'].tolist()
        if not relevant:
            continue

        # Rank semua loker berdasarkan final score
        ranked = (
            group.sort_values('final_score', ascending=False)
                 ['idlowongan'].tolist()
        )

        ndcg_list.append(ndcg_at_k(ranked, relevant, k))
        prec_list.append(precision_at_k(ranked, relevant, k))

    return {
        'ndcg':      round(np.mean(ndcg_list), 5),
        'precision': round(np.mean(prec_list), 5),
        'n_pelamar': len(ndcg_list)
    }


# Generate kombinasi bobot
print("Generating kombinasi bobot...")

weight_combinations = []
steps = [round(x, 2) for x in np.arange(0.05, 0.71, 0.05)]

for w1 in steps:
    for w2 in steps:
        for w3 in steps:
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if 0.05 <= w4 <= 0.70:
                weight_combinations.append((w1, w2, w3, w4))

print(f"Total kombinasi yang akan dicoba: {len(weight_combinations):,}")

# Jalankan ablation
print("\nMenjalankan ablation study...")
results = []

for w1, w2, w3, w4 in tqdm(weight_combinations):
    metrics = evaluate_weights(w1, w2, w3, w4, df_all)
    results.append({
        'w_semantic' : w1,
        'w_skill'    : w2,
        'w_edu'      : w3,
        'w_exp'      : w4,
        'ndcg'       : metrics['ndcg'],
        'precision'  : metrics['precision'],
    })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('ndcg', ascending=False)

print("\nAblation study selesai.")
print("\nTop 10 kombinasi bobot terbaik (berdasarkan NDCG@10):")
print(df_results.head(10).to_string(index=False))


# ============================================================ #
# CELL 11 — ANALISIS HASIL ABLATION                            #
# ============================================================ #

best = df_results.iloc[0]
print("\n" + "="*55)
print("BOBOT OPTIMAL")
print("="*55)
print(f"  Semantic  (w1) : {best['w_semantic']:.2f}")
print(f"  Skill     (w2) : {best['w_skill']:.2f}")
print(f"  Education (w3) : {best['w_edu']:.2f}")
print(f"  Experience(w4) : {best['w_exp']:.2f}")
print(f"  NDCG@10        : {best['ndcg']:.4f}")
print(f"  Precision@10   : {best['precision']:.4f}")

# Visualisasi: heatmap bobot terbaik
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: semantic vs skill bobot
top50 = df_results.head(50)
sc = axes[0].scatter(
    top50['w_semantic'], top50['w_skill'],
    c=top50['ndcg'], cmap='viridis', s=100, alpha=0.8
)
axes[0].set_xlabel('Bobot Semantic')
axes[0].set_ylabel('Bobot Skill')
axes[0].set_title('Top 50 Kombinasi: Semantic vs Skill (warna = NDCG)')
plt.colorbar(sc, ax=axes[0])
axes[0].scatter(
    best['w_semantic'], best['w_skill'],
    color='red', s=200, marker='*', label='Terbaik', zorder=5
)
axes[0].legend()

# Bar chart bobot optimal
komponen = ['Semantic', 'Skill', 'Education', 'Experience']
bobots   = [best['w_semantic'], best['w_skill'],
            best['w_edu'],      best['w_exp']]
bars = axes[1].bar(komponen, bobots, color=['steelblue','coral','green','orange'])
for bar, val in zip(bars, bobots):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
axes[1].set_ylim(0, max(bobots) + 0.1)
axes[1].set_title('Bobot Optimal per Komponen')
axes[1].set_ylabel('Bobot')

plt.tight_layout()
plt.savefig('ablation_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Sensitivity analysis: seberapa stabil bobot optimal?
print("\n--- Sensitivity Analysis ---")
print("Top 20 kombinasi bobot terbaik:")
print(df_results.head(20)[['w_semantic','w_skill','w_edu','w_exp','ndcg']].to_string(index=False))

print("\n[INTERPRETASI]")
print("Kalau top 20 kombinasi punya w_semantic yang mirip semua")
print("→ bobot semantic itu stabil dan reliable.")
print("Kalau sangat bervariasi → komponen itu tidak terlalu sensitif,")
print("perlu dicatat sebagai limitasi.")


# ============================================================ #
# CELL 12 — SIMPAN HASIL                                       #
# ============================================================ #

df_results.to_csv('ablation_results.csv', index=False)

# Simpan bobot optimal ke file terpisah
best_weights = {
    'w_semantic' : float(best['w_semantic']),
    'w_skill'    : float(best['w_skill']),
    'w_edu'      : float(best['w_edu']),
    'w_exp'      : float(best['w_exp']),
    'ndcg_at_10' : float(best['ndcg']),
    'precision_at_10': float(best['precision']),
    'catatan'    : (
        'Bobot ditentukan via ablation study menggunakan implicit feedback '
        '(pilihan melamar pelamar). Label mengandung selection bias dari '
        'konteks career day. Lihat Hu et al. (2008) untuk justifikasi '
        'pendekatan implicit feedback.'
    )
}

import json
with open('optimal_weights.json', 'w') as f:
    json.dump(best_weights, f, indent=2)

print("File tersimpan:")
print("  ablation_results.csv  → semua kombinasi bobot + skor")
print("  ablation_results.png  → visualisasi")
print("  score_distributions.png → distribusi score")
print("  optimal_weights.json  → bobot optimal final")


# ============================================================ #
# SQL UNTUK EXPORT CSV (jalankan di MySQL dulu)                #
# ============================================================ #
#
# -- VIEW 1: Skill agregasi
# CREATE OR REPLACE VIEW v_skill_agg AS
# SELECT idpelamar,
#   GROUP_CONCAT(CONCAT(namaskill,'|',keterangan)
#     ORDER BY namaskill SEPARATOR ';;') AS skills_detail,
#   GROUP_CONCAT(namaskill ORDER BY namaskill SEPARATOR ', ') AS skills_raw,
#   COUNT(*) AS jumlah_skill,
#   AVG(CASE keterangan WHEN 'Kurang' THEN 1 WHEN 'Cukup' THEN 2
#     WHEN 'Baik' THEN 3 WHEN 'Sangat Baik' THEN 4 END) AS avg_skill_level
# FROM pelamarskills GROUP BY idpelamar;
#
# -- VIEW 2: Pendidikan tertinggi
# CREATE OR REPLACE VIEW v_edu_agg AS
# SELECT idpelamar,
#   MAX(CASE kategori WHEN 'SD' THEN 1 WHEN 'SMP' THEN 2
#     WHEN 'SMA' THEN 3 WHEN 'SMK' THEN 3 WHEN 'D1' THEN 4
#     WHEN 'D2' THEN 4 WHEN 'D3' THEN 5 WHEN 'D4' THEN 6
#     WHEN 'S1' THEN 6 WHEN 'S2' THEN 8 WHEN 'S3' THEN 10
#     ELSE 0 END) AS edu_level_int,
#   SUBSTRING_INDEX(GROUP_CONCAT(kategori ORDER BY
#     CASE kategori WHEN 'S3' THEN 10 WHEN 'S2' THEN 8
#     WHEN 'S1' THEN 6 WHEN 'D4' THEN 6 WHEN 'D3' THEN 5
#     WHEN 'D2' THEN 4 WHEN 'D1' THEN 4 WHEN 'SMK' THEN 3
#     WHEN 'SMA' THEN 3 WHEN 'SMP' THEN 2 WHEN 'SD' THEN 1
#     ELSE 0 END DESC),',',1) AS pendidikan_tertinggi,
#   SUBSTRING_INDEX(GROUP_CONCAT(COALESCE(jurusan,'') ORDER BY
#     CASE kategori WHEN 'S3' THEN 10 WHEN 'S2' THEN 8
#     WHEN 'S1' THEN 6 WHEN 'D4' THEN 6 WHEN 'D3' THEN 5
#     WHEN 'D2' THEN 4 WHEN 'D1' THEN 4 WHEN 'SMK' THEN 3
#     WHEN 'SMA' THEN 3 WHEN 'SMP' THEN 2 WHEN 'SD' THEN 1
#     ELSE 0 END DESC),',',1) AS jurusan_tertinggi
# FROM pelamarpendidikans GROUP BY idpelamar;
#
# -- VIEW 3: Pengalaman
# CREATE OR REPLACE VIEW v_exp_agg AS
# SELECT idpelamar,
#   COUNT(*) AS jumlah_pengalaman,
#   GROUP_CONCAT(CONCAT(posisi,'(',tahunawal,'-',
#     IFNULL(tahunselesai,'skrg'),')') ORDER BY tahunawal DESC
#     SEPARATOR ';;') AS pengalaman_detail,
#   GROUP_CONCAT(posisi ORDER BY tahunawal DESC SEPARATOR ', ') AS posisi_all,
#   SUM(CASE WHEN aktif=1 THEN YEAR(CURDATE())-tahunawal
#     ELSE COALESCE(tahunselesai,YEAR(CURDATE()))-tahunawal END) AS total_tahun_exp,
#   MAX(aktif) AS sedang_bekerja
# FROM pelamarpengalamen GROUP BY idpelamar;
#
# -- DATASET UTAMA (export ke dataset_matching.csv)
# SELECT l.id AS lamaran_id, l.idpelamar, l.idlowongan,
#   l.ideven, l.statusditerima, l.tanggalmelamar,
#   p.namalengkap, p.jeniskelamin,
#   TIMESTAMPDIFF(YEAR,p.tanggallahir,CURDATE()) AS usia,
#   p.deskripsidiri,
#   sk.skills_detail, sk.skills_raw, sk.jumlah_skill,
#   ed.pendidikan_tertinggi, ed.edu_level_int, ed.jurusan_tertinggi,
#   ex.jumlah_pengalaman, ex.total_tahun_exp,
#   ex.pengalaman_detail, ex.posisi_all,
#   lo.namalowongan, lo.deskripsi AS deskripsi_loker,
#   lo.kategorilokasi, lo.gaji_awal, lo.gaji_akhir,
#   kl.id AS idkategori_loker, kl.nama AS kategori_loker
# FROM lamarans l
# JOIN pelamars p ON p.id=l.idpelamar
# JOIN lowongans lo ON lo.id=l.idlowongan
# JOIN kategorilowongans kl ON kl.id=lo.idkategorilowongan
# LEFT JOIN v_skill_agg sk ON sk.idpelamar=l.idpelamar
# LEFT JOIN v_edu_agg ed ON ed.idpelamar=l.idpelamar
# LEFT JOIN v_exp_agg ex ON ex.idpelamar=l.idpelamar;
#
# -- SEMUA LOWONGAN (export ke all_lowongans.csv)
# SELECT lo.id AS idlowongan, lo.namalowongan,
#   lo.deskripsi AS deskripsi_loker, lo.kategorilokasi,
#   lo.gaji_awal, lo.gaji_akhir, lo.status,
#   kl.id AS idkategori_loker, kl.nama AS kategori_loker
# FROM lowongans lo
# JOIN kategorilowongans kl ON kl.id=lo.idkategorilowongan;
#
# -- SEMUA PELAMAR (export ke all_pelamars.csv)
# SELECT p.id AS idpelamar, p.namalengkap, p.jeniskelamin,
#   p.deskripsidiri, sk.skills_detail, sk.skills_raw,
#   sk.jumlah_skill, ed.pendidikan_tertinggi, ed.edu_level_int,
#   ed.jurusan_tertinggi, ex.jumlah_pengalaman,
#   ex.total_tahun_exp, ex.pengalaman_detail, ex.posisi_all
# FROM pelamars p
# LEFT JOIN v_skill_agg sk ON sk.idpelamar=p.id
# LEFT JOIN v_edu_agg ed ON ed.idpelamar=p.id
# LEFT JOIN v_exp_agg ex ON ex.idpelamar=p.id;
