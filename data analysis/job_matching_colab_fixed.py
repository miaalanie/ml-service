# JOB MATCHING — ABLATION STUDY NOTEBOOK
# Tugas Akhir: Sistem Rekomendasi Lowongan Career Day

# Label: pilihan pelamar (implicit feedback)

# Limitasi: diakui sebagai proxy, bukan ground truth kecocokan

# Tujuan:
# 1. EDA — pahami karakteristik data
# 2. Generate negative samples
# 3. Hitung komponen score (semantic, skill, edu, exp)
# 4. Ablation study → bobot optimal
# 5. Evaluasi dengan Precision@K dan NDCG@K

import pandas as pd
import json
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

# Import selesai.
# ========================================================================

# Upload 3 file CSV dari MySQL:
# 1. dataset_matching.csv  → dari v_dataset_matching
# 2. all_lowongans.csv     → semua loker
# 3. all_pelamars.csv      → semua pelamar lengkap

from google.colab import drive
drive.mount('/content/drive')

# Untuk testing lokal:
df_lamaran  = pd.read_csv('/content/drive/MyDrive/Machine_Learning/TA/v_dataset_matching.csv')
df_lowongan = pd.read_csv('/content/drive/MyDrive/Machine_Learning/TA/v_lowongans.csv')
df_pelamar  = pd.read_csv('/content/drive/MyDrive/Machine_Learning/TA/v_pelamars.csv')

def validate_columns(df, required_cols, df_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  [ERROR] {df_name} — kolom tidak ada: {missing}")
        print(f"          Pastikan pakai SQL export versi terbaru (v3 LIKE-based).")
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

# Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
# Validasi kolom CSV:
#   [OK] dataset_matching.csv
#   [OK] all_lowongans.csv
#   [OK] all_pelamars.csv

# Ringkasan data:
#   Positive samples (lamaran) : 2,000 record
#   Lowongan unik              : 58
#   Pelamar unik               : 971
#   Kategori loker             : 23 kategori

# ========================================================================
df_lamaran.head()
# lamaran_id	idpelamar	idlowongan	ideven	tanggalmelamar	label	namalengkap	jeniskelamin	usia	skills_detail	...	pengalaman_detail	posisi_all	sedang_bekerja	namalowongan	deskripsi_loker	kategorilokasi	gaji_awal	gaji_akhir	idkategori_loker	kategori_loker
# 0	6	4	27	3	18/4/2026	1	Rifki Muh Sopiandi	Laki-laki	26	Exel, Wor, PPT, Sap R3|Sangat Baik	...	Team Leader Dept Continues Improvement(2018-2026)	Team Leader Dept Continues Improvement	1.0	Asisten Supervisor Pabrik	<ul><li>Membantu dalam pelaksanaan rencana pro...	Dalam Negeri	6000000.0	8000000.0	70	Personal Assistant / Secretary
# 1	7	4	25	3	18/4/2026	1	Rifki Muh Sopiandi	Laki-laki	26	Exel, Wor, PPT, Sap R3|Sangat Baik	...	Team Leader Dept Continues Improvement(2018-2026)	Team Leader Dept Continues Improvement	1.0	Spesialis Sumber Daya Manusia	<p><strong>Tanggung Jawab Pekerjaan:</strong><...	Dalam Negeri	6000000.0	8000000.0	68	Recruiter / Talent Acquisition
# 2	8	4	26	3	18/4/2026	1	Rifki Muh Sopiandi	Laki-laki	26	Exel, Wor, PPT, Sap R3|Sangat Baik	...	Team Leader Dept Continues Improvement(2018-2026)	Team Leader Dept Continues Improvement	1.0	Spesialis Peralatan	<p><strong>Tanggung Jawab Pekerjaan:</strong><...	Dalam Negeri	6000000.0	8000000.0	80	Quality Control (QC) Inspector
# 3	9	10	37	3	20/4/2026	1	REGY GLASIO C.PRATAMA	Laki-laki	20	Mengoperasikan mesin cnc/forklift|Cukup	...	Operator produksi(2025-2025)	Operator produksi	1.0	Operator Mesin (Machine Operator)	<h4><strong>Tugas &amp; Tanggung Jawab</strong...	Dalam Negeri	NaN	NaN	37	Mechanical Engineer
# 4	10	11	15	3	20/4/2026	1	M.SAMSUL FIKRI	Laki-laki	18	Desain gerfis Corel draw and Canva|Baik;;Editi...	...	Magang(2023-2026)	Magang	1.0	Marketing	<ul><li>Memiliki pemahaman yang baik mengenai ...	Dalam Negeri	10000000.0	15000000.0	20	Marketing Executive
# 5 rows × 28 columns

df_lowongan.head()
[{"index":0,"idlowongan":13,"namalowongan":"Admin Proyek","deskripsi_loker":"<p><strong>Tanggung Jawab Pekerjaan:</strong></p><ul><li>Memberikan dukungan administratif kepada tim proyek untuk memastikan pelaksanaan berjalan lancar;</li><li>Membuat sales order;</li><li>Memelihara dan memperbarui dokumentasi proyek, jadwal, serta laporan;</li><li>Mengkoordinasikan rapat, mengelola jadwal, serta menangani komunikasi terkait proyek;</li><li>Memantau timeline proyek serta mengidentifikasi potensi risiko atau keterlambatan;</li><li>Berkolaborasi dengan tim lintas fungsi untuk mencapai tujuan proyek.</li></ul><p><strong>Kualifikasi Pekerjaan:</strong></p><ul><li>Minimal lulusan S1 dari semua jurusan (diutamakan Manajemen, Bisnis, atau Keuangan);</li><li>Wajib fasih berbahasa Mandarin (lisan dan tulisan) dan mampu menggunakannya sebagai bahasa kerja;</li><li>Menguasai Microsoft Office (Excel, PowerPoint, dan Word);</li><li>Memiliki minat di bidang branding, administrasi pemasaran, dan komunikasi publik;</li><li>Memiliki kemampuan manajemen, manajemen waktu, serta jiwa kepemimpinan yang baik;</li><li>Memiliki kemampuan komunikasi yang baik, mampu multitasking, serta berkepribadian terbuka;</li><li>Bersedia melakukan perjalanan dinas;</li><li>Terampil menggunakan Microsoft Office (Excel, Word, dan PowerPoint).</li></ul>","kategorilokasi":"Dalam Negeri","gaji_awal":"6000000.0","gaji_akhir":"7500000.0","status_loker":1,"kuota":20,"idkategori_loker":71,"kategori_loker":"Administrative Staff"},{"index":1,"idlowongan":14,"namalowongan":"Staf Penjualan / Merchandiser","deskripsi_loker":"<p><strong>Tanggung Jawab Pekerjaan:</strong></p><ul><li>Mengembangkan pelanggan baru sekaligus menjaga hubungan dengan pelanggan yang sudah ada untuk meningkatkan kepuasan dan loyalitas;</li><li>Bertanggung jawab atas analisis kebutuhan sampel, pengembangan desain, serta pengelolaan penawaran harga;</li><li>Mengawasi pelaksanaan produksi dan mengendalikan seluruh proses pesanan;</li><li>Mengoordinasikan jadwal produksi;</li><li>Melakukan evaluasi terhadap standar pengujian produk;</li><li>Memantau dan menelusuri perkembangan pesanan;</li><li>Menjaga komunikasi yang efektif dengan tim internal dan pelanggan.</li></ul><p><strong>Kualifikasi Pekerjaan:</strong></p><ul><li>Fasih berbahasa Mandarin, Inggris, dan Indonesia, baik lisan maupun tulisan;</li><li>Memiliki kemampuan administrasi dan tindak lanjut pesanan yang baik;</li><li>Mampu bekerja secara teliti serta menangani banyak tugas (multitasking);</li><li>Terbuka untuk lulusan baru (fresh graduate);</li><li>Memiliki pengalaman di bidang penjualan, merchandising, penanganan pelanggan, atau manajemen pesanan menjadi nilai tambah.</li></ul>","kategorilokasi":"Dalam Negeri","gaji_awal":"15000000.0","gaji_akhir":"20000000.0","status_loker":1,"kuota":10,"idkategori_loker":21,"kategori_loker":"Sales / Account Executive"},{"index":2,"idlowongan":15,"namalowongan":"Marketing","deskripsi_loker":"<ul><li>Memiliki pemahaman yang baik mengenai wilayah Jakarta, Bandung, dan sekitarnya, dengan fokus pada pengembangan kerja sama dengan pabrik pemintalan kapas dan pabrik benang, membangun basis data pelanggan, serta mengembangkan dan menjaga hubungan yang kuat dengan klien;</li><li>Mengemudikan kendaraan perusahaan untuk mendampingi manajer dan engineer dari Tiongkok saat mengunjungi pelanggan, memberikan dukungan koordinasi dengan pelanggan, serta membantu mencapai target penjualan perusahaan;</li><li>Wajib memiliki SIM yang masih berlaku dan mampu mengemudi dengan baik;</li><li>Mengenal wilayah Jakarta, Bandung, serta kawasan industri di sekitarnya;</li><li>Memiliki kemampuan komunikasi bisnis dan koordinasi yang baik;</li><li>Fasih berbahasa Mandarin (lisan dan tulisan);</li><li>Memiliki minat dan semangat di bidang penjualan dan pemasaran;</li><li>Mampu bekerja secara efektif dalam tim;</li><li>Diutamakan memiliki pengetahuan di industri tekstil.</li></ul>","kategorilokasi":"Dalam Negeri","gaji_awal":"10000000.0","gaji_akhir":"15000000.0","status_loker":1,"kuota":20,"idkategori_loker":20,"kategori_loker":"Marketing Executive"},{"index":3,"idlowongan":16,"namalowongan":"Pemasaran Penjualan","deskripsi_loker":"<p><strong>Tanggung Jawab Pekerjaan:</strong></p><ul><li>Mengidentifikasi dan mengembangkan peluang bisnis baru di pasar yang ditargetkan;</li><li>Membangun, menjaga, dan memperkuat hubungan jangka panjang dengan pelanggan serta mitra bisnis;</li><li>Melakukan riset pasar dan analisis kompetitor untuk memahami kebutuhan pelanggan;</li><li>Mempresentasikan produk, solusi, dan layanan terkait mesin kepada calon pelanggan;</li><li>Berkoordinasi dengan tim teknis dan engineering untuk memastikan kebutuhan pelanggan terpenuhi;</li><li>Menyusun penawaran harga (quotation), proposal, serta laporan penjualan;</li><li>Menindaklanjuti pertanyaan pelanggan, proses negosiasi, serta memberikan dukungan purna jual;</li><li>Mendukung strategi penjualan untuk mencapai target pendapatan perusahaan;</li><li>Melakukan kunjungan rutin ke pelanggan dan lokasi industri untuk memahami penggunaan mesin serta kebutuhan operasional.</li></ul><p><strong>Kualifikasi Pekerjaan:</strong></p><ul><li>Minimal lulusan S1 di bidang Pemasaran, Bisnis, Teknik, Teknik Mesin, Teknik Industri, atau bidang terkait;</li><li>Memiliki pengalaman di bidang penjualan dan pemasaran lebih diutamakan;</li><li>Memahami mesin industri atau permesinan menjadi nilai tambah;</li><li>Memiliki kemampuan komunikasi, negosiasi, dan presentasi yang baik;</li><li>Berorientasi pada target serta mampu bekerja secara mandiri maupun dalam tim;</li><li>Memiliki kemampuan pemecahan masalah dan penanganan pelanggan yang baik;</li><li>Memahami konsep penjualan B2B dan pengelolaan pelanggan industri menjadi nilai tambah;</li><li>Bersedia melakukan perjalanan dinas untuk kunjungan pelanggan dan pengembangan bisnis.</li></ul>","kategorilokasi":"Dalam Negeri","gaji_awal":"4000000.0","gaji_akhir":"10000000.0","status_loker":1,"kuota":10,"idkategori_loker":21,"kategori_loker":"Sales / Account Executive"},{"index":4,"idlowongan":17,"namalowongan":"Teknisi Mesin","deskripsi_loker":"<p><strong>Tanggung Jawab Pekerjaan:</strong></p><p>Memahami serta menangani permasalahan mesin mekanik dan kelistrikan, khususnya yang berkaitan dengan:</p><ul><li>Genset diesel.</li><li>Unit generator gas.</li><li>Sistem pompa tenaga.</li><li>Peralatan tenaga kelautan.</li><li>Sistem kontrol otomatis.</li></ul><ol><li>Melakukan instalasi, inspeksi, perawatan preventif, serta perbaikan mesin dan komponen kelistrikan;</li><li>Mendiagnosis kerusakan pada mesin, generator, pompa, dan panel kontrol;</li><li>Membantu proses pengujian mesin, commissioning, serta pengecekan operasional sebelum pengiriman atau pemasangan;</li><li>Memantau kinerja mesin untuk memastikan operasi yang stabil, aman, dan efisien;</li><li>Bekerja sama dengan tim engineer dan service untuk menyelesaikan permasalahan teknis;</li><li>Menyusun laporan servis, catatan perawatan, serta penggunaan suku cadang;</li><li>Mendukung penanganan gangguan darurat di lokasi pelanggan bila diperlukan.</li></ol><p><strong>Kualifikasi Pekerjaan:</strong></p><p>Minimal lulusan S1 atau D3 di bidang:</p><ul><li>Teknik Mesin.</li><li>Teknik Elektro.</li><li>Mekatronika.</li><li>Teknik Industri.</li><li>Atau bidang teknis terkait.</li></ul><p>Memiliki pemahaman yang baik mengenai mesin dan troubleshooting kelistrikan;<br>Familiar dengan:</p><ul><li>Mesin diesel.</li><li>Sistem generator.</li><li>Sistem pompa.</li><li>PLC / panel kontrol.</li><li>Sistem kelistrikan industri.</li></ul><ol><li>Diutamakan memiliki pengalaman di bidang generator, alat berat, manufaktur, kelautan, atau mesin industri;</li><li>Mampu membaca gambar teknik, diagram kelistrikan, serta manual mesin;</li><li>Memiliki kemampuan analisis dan pemecahan masalah yang baik;</li><li>Bersedia melakukan perjalanan dinas untuk perawatan di lokasi dan dukungan pelanggan;</li><li>Mampu bekerja secara mandiri maupun dalam tim.</li></ol>","kategorilokasi":"Dalam Negeri","gaji_awal":"4000000.0","gaji_akhir":"7000000.0","status_loker":1,"kuota":15,"idkategori_loker":37,"kategori_loker":"Mechanical Engineer"}]
df_pelamar.head()
[{"index":0,"idpelamar":2,"namalengkap":"Lia","jeniskelamin":"Perempuan","usia":5,"skills_detail":"Microsoft Excel|Baik","skills_raw":"Microsoft Excel","jumlah_skill":1,"avg_skill_level":"3.0","pendidikan_tertinggi":"D4/S1","edu_level_int":6,"jurusan_tertinggi":"Teknik Informatika","jumlah_pengalaman":"NaN","total_tahun_exp":"NaN","pengalaman_detail":"NaN","posisi_all":"NaN","sedang_bekerja":"NaN"},{"index":1,"idpelamar":3,"namalengkap":"Fathir Galih Alamsyah","jeniskelamin":"Laki-laki","usia":18,"skills_detail":"Instalasi Listrik, Maintenance|Baik","skills_raw":"Instalasi Listrik, Maintenance","jumlah_skill":1,"avg_skill_level":"3.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"Teknik Instalasi Tenaga Listrik","jumlah_pengalaman":"1.0","total_tahun_exp":"1.0","pengalaman_detail":"Magang(2025-2025)","posisi_all":"Magang","sedang_bekerja":"1.0"},{"index":2,"idpelamar":4,"namalengkap":"Rifki Muh Sopiandi","jeniskelamin":"Laki-laki","usia":26,"skills_detail":"Exel, Wor, PPT, Sap R3|Sangat Baik","skills_raw":"Exel, Wor, PPT, Sap R3","jumlah_skill":1,"avg_skill_level":"4.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"TKJ","jumlah_pengalaman":"1.0","total_tahun_exp":"8.0","pengalaman_detail":"Team Leader Dept Continues Improvement(2018-2026)","posisi_all":"Team Leader Dept Continues Improvement","sedang_bekerja":"1.0"},{"index":3,"idpelamar":5,"namalengkap":"Nisrina Meilani","jeniskelamin":"Perempuan","usia":22,"skills_detail":"Microsoft Windows|Kurang","skills_raw":"Microsoft Windows","jumlah_skill":1,"avg_skill_level":"1.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"Farmasi","jumlah_pengalaman":"1.0","total_tahun_exp":"1.0","pengalaman_detail":"Operator Produksi(2025-2025)","posisi_all":"Operator Produksi","sedang_bekerja":"1.0"},{"index":4,"idpelamar":6,"namalengkap":"Andi kusnadi","jeniskelamin":"Laki-laki","usia":27,"skills_detail":"Mengoperasikan mesin stamping|Baik","skills_raw":"Mengoperasikan mesin stamping","jumlah_skill":1,"avg_skill_level":"3.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"Teknik otomotif sepeda motor","jumlah_pengalaman":"1.0","total_tahun_exp":"6.0","pengalaman_detail":"Operator produksi(2020-2026)","posisi_all":"Operator produksi","sedang_bekerja":"1.0"}]

# ========================================================================
# EDA
def _edu_level_from_str(kategori_str):
    """
    Konversi string kategori pendidikan ke skor ordinal.

    Pakai str.contains() (via 'in') — bukan exact match —
    karena DB menyimpan "D4/S1", "SMA/SMK" dll.

    Order pengecekan dari tertinggi ke terendah supaya tidak
    salah: "D4/S1" cek S3 (F), S2 (F), S1 (T) → 6 ✓
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

print("\n" + "="*60)
print("EDA — KARAKTERISTIK DATA")
print("="*60)

# 3a. Kelengkapan profil pelamar
print("\n--- 3a. Kelengkapan Profil Pelamar ---")
cols_check = {
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

# 3b. Cek nilai unik pendidikan_tertinggi dari data real
print("\n--- 3b. Nilai Unik Pendidikan Tertinggi (dari data real) ---")
if 'pendidikan_tertinggi' in df_pelamar.columns:
    uniq_edu = df_pelamar['pendidikan_tertinggi'].value_counts()
    for k, v in uniq_edu.items():
        # FIX-5: sekarang _edu_level_from_str sudah terdefinisi di sini
        score = _edu_level_from_str(str(k))
        print(f"  '{k}' : {v} pelamar → level {score}")
    print(f"\n  [NOTE] Format dari DB bisa 'D4/S1', 'SMA/SMK' dll.")
    print(f"  _edu_level_from_str() pakai 'in' check — sudah handle format ini.")

# 3c. Cek edu_level_int dari SQL
print("\n--- 3c. Distribusi edu_level_int (dari SQL view) ---")
if 'edu_level_int' in df_pelamar.columns:
    lv_dist = df_pelamar['edu_level_int'].value_counts().sort_index()
    print(f"  Nilai 0 (tidak terdeteksi) : {(df_pelamar['edu_level_int']==0).sum()} pelamar")
    for k, v in lv_dist.items():
        print(f"  Level {k:2} : {v} pelamar")
    if (df_pelamar['edu_level_int'] == 0).all():
        print("\n  [WARNING] Semua edu_level_int masih 0!")
        print("  Pastikan SQL view sudah diupdate ke versi LIKE-based (v3).")

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
        # FIX-1 preview: pattern yang benar (greedy dari kanan)
        pattern = re.compile(r'^(.+)\((\d{4})-([\d]+|skrg)\)$')
        valid = sample_exp.apply(
            lambda x: all(pattern.match(s.strip()) for s in str(x).split(';;'))
        ).sum()
        print(f"  Format valid (regex match): {valid}/{len(sample_exp)}")
    else:
        print("  Tidak ada pelamar dengan pengalaman kerja.")


# ============================================================
# EDA — KARAKTERISTIK DATA
# ============================================================

# --- 3a. Kelengkapan Profil Pelamar ---
#   Ada skill                      : 100.0%
#   Ada skill detail               : 100.0%
#   Ada pendidikan                 : 100.0%
#   Ada jurusan                    : 99.0%
#   Ada pengalaman                 : 64.7%

# --- 3b. Nilai Unik Pendidikan Tertinggi (dari data real) ---
#   'SMA/SMK' : 606 pelamar → level 3
#   'D4/S1' : 281 pelamar → level 6
#   'D3' : 71 pelamar → level 5
#   'SMP' : 6 pelamar → level 2
#   'SD' : 4 pelamar → level 1
#   'S2' : 3 pelamar → level 8

#   [NOTE] Format dari DB bisa 'D4/S1', 'SMA/SMK' dll.
#   _edu_level_from_str() pakai 'in' check — sudah handle format ini.

# --- 3c. Distribusi edu_level_int (dari SQL view) ---
#   Nilai 0 (tidak terdeteksi) : 0 pelamar
#   Level  1 : 4 pelamar
#   Level  2 : 6 pelamar
#   Level  3 : 606 pelamar
#   Level  5 : 71 pelamar
#   Level  6 : 281 pelamar
#   Level  8 : 3 pelamar

# --- 3d. Distribusi Lamaran per Pelamar ---
#   Rata-rata : 3.0
#   Median    : 3
#   Min - Max : 1 - 5
#   Melamar 1 loker  : 163 pelamar
#   Melamar >3 loker : 256 pelamar

# --- 3e. Distribusi Kategori Loker ---
#   Sales / Account Executive                : 15 loker
#   Administrative Staff                     : 9 loker
#   Mechanical Engineer                      : 6 loker
#   Translator / Interpreter                 : 3 loker
#   Recruiter / Talent Acquisition           : 2 loker
#   HR Generalist / Manager                  : 2 loker
#   Software Engineer                        : 2 loker
#   Accounting                               : 2 loker
#   Customer Service / Call Center           : 2 loker
#   Finance / Treasury                       : 2 loker
#   Marketing Executive                      : 1 loker
#   Legal Counsel / Lawyer                   : 1 loker
#   Tax Specialist                           : 1 loker
#   Graphic Designer                         : 1 loker
#   Electrical Engineer                      : 1 loker
#   Quality Control (QC) Inspector           : 1 loker
#   Personal Assistant / Secretary           : 1 loker
#   Technical Support                        : 1 loker
#   Network & System Administrator           : 1 loker
#   Office Manager                           : 1 loker
#   Retail Sales Associate                   : 1 loker
#   Production Operator / Assembly           : 1 loker
#   Logistics / Supply Chain Manager         : 1 loker

# --- 3f. Verifikasi Format skills_detail ---
#   Contoh 1: Microsoft Excel|Baik
#   Contoh 2: Instalasi Listrik, Maintenance|Baik
#   Punya multi-skill (ada ;;) : 315 pelamar
#   Single skill saja          : 656 pelamar

# --- 3g. Verifikasi Format pengalaman_detail ---
#   Contoh: Magang(2025-2025)
#   Format valid (regex match): 628/628

# ========================================================================
# PREPROCESSING TEKS
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
    Kolom: skills_raw, pendidikan_tertinggi,
           jurusan_tertinggi, posisi_all
    """
    parts = []
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

# ✓ Preprocessing selesai.
#   Pelamar tanpa teks  : 0 (fallback score 0)
#   Lowongan tanpa teks : 0

#   Contoh pelamar_text :
#   keahlian yang dimiliki: microsoft excel latar belakang pendidikan d4/s1 jurusan teknik informatika

#   Contoh lowongan_text:
#   posisi yang dibutuhkan: admin proyek kategori pekerjaan: administrative staff deskripsi pekerjaan: tanggung jawab pekerjaan: memberikan dukungan administratif kepada tim proyek untuk memastikan pelaks

# ========================================================================
# EMBEDDING
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

# Loading model...
# Loading weights: 100%
#  199/199 [00:00<00:00, 673.27it/s, Materializing param=pooler.dense.weight]
# BertModel LOAD REPORT from: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# Key                     | Status     |  | 
# ------------------------+------------+--+-
# embeddings.position_ids | UNEXPECTED |  | 

# Notes:
# - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# ✓ Model loaded.

# Encoding 971 profil pelamar...
# Batches: 100%
#  16/16 [00:31<00:00,  1.14it/s]

# Encoding 58 lowongan...
# Batches: 100%
#  1/1 [00:05<00:00,  5.15s/it]

# ✓ Embedding selesai. Dimensi: 384

# ========================================================================
# GENERATE NEGATIVE SAMPLES
# Strategi: untuk setiap pelamar, ambil loker di kategori yang
# SAMA yang tidak dia lamar → label = 0
# Ratio 1:2 (positif:negatif)

print("Generating negative samples...")

if 'idkategori_loker' not in df_lamaran.columns:
    raise ValueError(
        "Kolom 'idkategori_loker' tidak ada di dataset_matching.csv.\n"
        "Pastikan pakai SQL export versi terbaru (v3 LIKE-based)."
    )

# FIX-3: Pastikan lowongan_text sudah ada di df_lowongan
# (Cell 4 harus sudah jalan sebelum ini)
assert 'lowongan_text' in df_lowongan.columns, (
    "Kolom 'lowongan_text' belum ada di df_lowongan. "
    "Jalankan Cell 4 (Preprocessing) sebelum Cell 6."
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
    n_negatif  = n_positif * NEGATIVE_RATIO
    replace_flag = len(candidate_negatives) < n_negatif
    idx_chosen = np.random.choice(len(candidate_negatives), size=n_negatif, replace=replace_flag)
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
            # FIX-3: lowongan_text dijamin ada karena Cell 4 sudah build-nya
            'lowongan_text'   : loker_data.get('lowongan_text', ''),
            'label'           : 0,
            'lamaran_id'      : None,
        })
        negative_samples.append(neg)

# FIX-4: Pastikan df_positive punya semua kolom yang dibutuhkan
# (skills, edu, exp, teks profil, teks loker)
df_positive = df_lamaran.copy()
df_positive['label'] = 1

# Kolom dari profil pelamar (skills, edu, exp) — join ke df_pelamar
COLS_FROM_PELAMAR = [
    'idpelamar', 'skills_raw', 'jumlah_skill',
    'avg_skill_level', 'pendidikan_tertinggi', 'edu_level_int',
    'jurusan_tertinggi', 'jumlah_pengalaman', 'total_tahun_exp',
    'pengalaman_detail', 'posisi_all', 'sedang_bekerja', 'pelamar_text',
]
missing_from_pelamar = [c for c in COLS_FROM_PELAMAR
                        if c != 'idpelamar' and c not in df_positive.columns]
if missing_from_pelamar:
    avail = [c for c in COLS_FROM_PELAMAR if c in df_pelamar.columns]
    df_positive = df_positive.merge(
        df_pelamar[avail], on='idpelamar', how='left', suffixes=('', '_pelamar')
    )
    print(f"  Merged dari df_pelamar: {missing_from_pelamar}")

# Kolom teks lowongan
if 'lowongan_text' not in df_positive.columns:
    df_positive = df_positive.merge(
        df_lowongan[['idlowongan', 'lowongan_text']], on='idlowongan', how='left'
    )
    print(f"  Merged lowongan_text dari df_lowongan")

df_negative = pd.DataFrame(negative_samples)
df_all      = pd.concat([df_positive, df_negative], ignore_index=True)

print(f"\n✓ Negative sampling selesai.")
print(f"  Positive : {len(df_positive):,}")
print(f"  Negative : {len(df_negative):,}")
print(f"  Total    : {len(df_all):,}")
ratio = len(df_negative) / max(len(df_positive), 1)
print(f"  Ratio    : 1:{ratio:.2f}")

# Generating negative samples...
# Generate negatives: 100%|██████████| 676/676 [00:01<00:00, 341.29it/s]  Merged dari df_pelamar: ['pelamar_text']
#   Merged lowongan_text dari df_lowongan

# ✓ Negative sampling selesai.
#   Positive : 2,000
#   Negative : 3,904
#   Total    : 5,904
#   Ratio    : 1:1.95

df_negative.head()
[{"index":0,"idpelamar":4,"namalengkap":"Rifki Muh Sopiandi","jeniskelamin":"Laki-laki","usia":26,"skills_detail":"Exel, Wor, PPT, Sap R3|Sangat Baik","skills_raw":"Exel, Wor, PPT, Sap R3","jumlah_skill":1,"avg_skill_level":"4.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"TKJ","jumlah_pengalaman":"1.0","total_tahun_exp":"8.0","pengalaman_detail":"Team Leader Dept Continues Improvement(2018-2026)","posisi_all":"Team Leader Dept Continues Improvement","sedang_bekerja":"1.0","pelamar_text":"keahlian yang dimiliki: exel, wor, ppt, sap r3 latar belakang pendidikan sma/smk jurusan tkj pengalaman kerja sebagai: team leader dept continues improvement","idlowongan":63,"namalowongan":"HR Supervisor","deskripsi_loker":"<h4>Tugas &amp; Tanggung Jawab</h4><ul><li>Melakukan supervisi terhadap seluruh aktivitas dan dinamika di departemen HR</li><li>Mengawasi proses rekrutmen, onboarding, dan pengelolaan karyawan</li><li>Mengelola administrasi HR (absensi, kontrak, payroll, dan data karyawan)</li><li>Memastikan kepatuhan terhadap peraturan ketenagakerjaan yang berlaku</li><li>Mengembangkan dan mengimplementasikan kebijakan serta SOP HR</li><li>Menangani hubungan industrial dan permasalahan karyawan</li><li>Melakukan evaluasi kinerja karyawan dan mendukung program pengembangan SDM</li><li>Menyusun laporan HR secara berkala kepada manajemen</li><li>Berkoordinasi dengan departemen lain terkait kebutuhan SDM</li></ul><h4>&nbsp;</h4><h4><strong>Kualifikasi</strong></h4><ul><li>Pendidikan minimal S1 (diutamakan Psikologi, Manajemen SDM, atau terkait)</li><li>Memiliki pengalaman minimal 2–4 tahun di bidang HR (pengalaman sebagai Supervisor menjadi nilai tambah)</li><li>Memahami proses HR secara menyeluruh (rekrutmen, payroll, hubungan industrial, dll.)</li><li>Memiliki kemampuan leadership dan komunikasi yang baik</li><li>Teliti, tegas, dan mampu mengambil keputusan</li><li>Mampu mengoperasikan Microsoft Office</li><li><strong>Mampu berbahasa Mandarin menjadi nilai tambah</strong></li></ul>"},{"index":1,"idpelamar":4,"namalengkap":"Rifki Muh Sopiandi","jeniskelamin":"Laki-laki","usia":26,"skills_detail":"Exel, Wor, PPT, Sap R3|Sangat Baik","skills_raw":"Exel, Wor, PPT, Sap R3","jumlah_skill":1,"avg_skill_level":"4.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"TKJ","jumlah_pengalaman":"1.0","total_tahun_exp":"8.0","pengalaman_detail":"Team Leader Dept Continues Improvement(2018-2026)","posisi_all":"Team Leader Dept Continues Improvement","sedang_bekerja":"1.0","pelamar_text":"keahlian yang dimiliki: exel, wor, ppt, sap r3 latar belakang pendidikan sma/smk jurusan tkj pengalaman kerja sebagai: team leader dept continues improvement","idlowongan":63,"namalowongan":"HR Supervisor","deskripsi_loker":"<h4>Tugas &amp; Tanggung Jawab</h4><ul><li>Melakukan supervisi terhadap seluruh aktivitas dan dinamika di departemen HR</li><li>Mengawasi proses rekrutmen, onboarding, dan pengelolaan karyawan</li><li>Mengelola administrasi HR (absensi, kontrak, payroll, dan data karyawan)</li><li>Memastikan kepatuhan terhadap peraturan ketenagakerjaan yang berlaku</li><li>Mengembangkan dan mengimplementasikan kebijakan serta SOP HR</li><li>Menangani hubungan industrial dan permasalahan karyawan</li><li>Melakukan evaluasi kinerja karyawan dan mendukung program pengembangan SDM</li><li>Menyusun laporan HR secara berkala kepada manajemen</li><li>Berkoordinasi dengan departemen lain terkait kebutuhan SDM</li></ul><h4>&nbsp;</h4><h4><strong>Kualifikasi</strong></h4><ul><li>Pendidikan minimal S1 (diutamakan Psikologi, Manajemen SDM, atau terkait)</li><li>Memiliki pengalaman minimal 2–4 tahun di bidang HR (pengalaman sebagai Supervisor menjadi nilai tambah)</li><li>Memahami proses HR secara menyeluruh (rekrutmen, payroll, hubungan industrial, dll.)</li><li>Memiliki kemampuan leadership dan komunikasi yang baik</li><li>Teliti, tegas, dan mampu mengambil keputusan</li><li>Mampu mengoperasikan Microsoft Office</li><li><strong>Mampu berbahasa Mandarin menjadi nilai tambah</strong></li></ul>"},{"index":2,"idpelamar":4,"namalengkap":"Rifki Muh Sopiandi","jeniskelamin":"Laki-laki","usia":26,"skills_detail":"Exel, Wor, PPT, Sap R3|Sangat Baik","skills_raw":"Exel, Wor, PPT, Sap R3","jumlah_skill":1,"avg_skill_level":"4.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"TKJ","jumlah_pengalaman":"1.0","total_tahun_exp":"8.0","pengalaman_detail":"Team Leader Dept Continues Improvement(2018-2026)","posisi_all":"Team Leader Dept Continues Improvement","sedang_bekerja":"1.0","pelamar_text":"keahlian yang dimiliki: exel, wor, ppt, sap r3 latar belakang pendidikan sma/smk jurusan tkj pengalaman kerja sebagai: team leader dept continues improvement","idlowongan":63,"namalowongan":"HR Supervisor","deskripsi_loker":"<h4>Tugas &amp; Tanggung Jawab</h4><ul><li>Melakukan supervisi terhadap seluruh aktivitas dan dinamika di departemen HR</li><li>Mengawasi proses rekrutmen, onboarding, dan pengelolaan karyawan</li><li>Mengelola administrasi HR (absensi, kontrak, payroll, dan data karyawan)</li><li>Memastikan kepatuhan terhadap peraturan ketenagakerjaan yang berlaku</li><li>Mengembangkan dan mengimplementasikan kebijakan serta SOP HR</li><li>Menangani hubungan industrial dan permasalahan karyawan</li><li>Melakukan evaluasi kinerja karyawan dan mendukung program pengembangan SDM</li><li>Menyusun laporan HR secara berkala kepada manajemen</li><li>Berkoordinasi dengan departemen lain terkait kebutuhan SDM</li></ul><h4>&nbsp;</h4><h4><strong>Kualifikasi</strong></h4><ul><li>Pendidikan minimal S1 (diutamakan Psikologi, Manajemen SDM, atau terkait)</li><li>Memiliki pengalaman minimal 2–4 tahun di bidang HR (pengalaman sebagai Supervisor menjadi nilai tambah)</li><li>Memahami proses HR secara menyeluruh (rekrutmen, payroll, hubungan industrial, dll.)</li><li>Memiliki kemampuan leadership dan komunikasi yang baik</li><li>Teliti, tegas, dan mampu mengambil keputusan</li><li>Mampu mengoperasikan Microsoft Office</li><li><strong>Mampu berbahasa Mandarin menjadi nilai tambah</strong></li></ul>"},{"index":3,"idpelamar":4,"namalengkap":"Rifki Muh Sopiandi","jeniskelamin":"Laki-laki","usia":26,"skills_detail":"Exel, Wor, PPT, Sap R3|Sangat Baik","skills_raw":"Exel, Wor, PPT, Sap R3","jumlah_skill":1,"avg_skill_level":"4.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"TKJ","jumlah_pengalaman":"1.0","total_tahun_exp":"8.0","pengalaman_detail":"Team Leader Dept Continues Improvement(2018-2026)","posisi_all":"Team Leader Dept Continues Improvement","sedang_bekerja":"1.0","pelamar_text":"keahlian yang dimiliki: exel, wor, ppt, sap r3 latar belakang pendidikan sma/smk jurusan tkj pengalaman kerja sebagai: team leader dept continues improvement","idlowongan":63,"namalowongan":"HR Supervisor","deskripsi_loker":"<h4>Tugas &amp; Tanggung Jawab</h4><ul><li>Melakukan supervisi terhadap seluruh aktivitas dan dinamika di departemen HR</li><li>Mengawasi proses rekrutmen, onboarding, dan pengelolaan karyawan</li><li>Mengelola administrasi HR (absensi, kontrak, payroll, dan data karyawan)</li><li>Memastikan kepatuhan terhadap peraturan ketenagakerjaan yang berlaku</li><li>Mengembangkan dan mengimplementasikan kebijakan serta SOP HR</li><li>Menangani hubungan industrial dan permasalahan karyawan</li><li>Melakukan evaluasi kinerja karyawan dan mendukung program pengembangan SDM</li><li>Menyusun laporan HR secara berkala kepada manajemen</li><li>Berkoordinasi dengan departemen lain terkait kebutuhan SDM</li></ul><h4>&nbsp;</h4><h4><strong>Kualifikasi</strong></h4><ul><li>Pendidikan minimal S1 (diutamakan Psikologi, Manajemen SDM, atau terkait)</li><li>Memiliki pengalaman minimal 2–4 tahun di bidang HR (pengalaman sebagai Supervisor menjadi nilai tambah)</li><li>Memahami proses HR secara menyeluruh (rekrutmen, payroll, hubungan industrial, dll.)</li><li>Memiliki kemampuan leadership dan komunikasi yang baik</li><li>Teliti, tegas, dan mampu mengambil keputusan</li><li>Mampu mengoperasikan Microsoft Office</li><li><strong>Mampu berbahasa Mandarin menjadi nilai tambah</strong></li></ul>"},{"index":4,"idpelamar":4,"namalengkap":"Rifki Muh Sopiandi","jeniskelamin":"Laki-laki","usia":26,"skills_detail":"Exel, Wor, PPT, Sap R3|Sangat Baik","skills_raw":"Exel, Wor, PPT, Sap R3","jumlah_skill":1,"avg_skill_level":"4.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"TKJ","jumlah_pengalaman":"1.0","total_tahun_exp":"8.0","pengalaman_detail":"Team Leader Dept Continues Improvement(2018-2026)","posisi_all":"Team Leader Dept Continues Improvement","sedang_bekerja":"1.0","pelamar_text":"keahlian yang dimiliki: exel, wor, ppt, sap r3 latar belakang pendidikan sma/smk jurusan tkj pengalaman kerja sebagai: team leader dept continues improvement","idlowongan":63,"namalowongan":"HR Supervisor","deskripsi_loker":"<h4>Tugas &amp; Tanggung Jawab</h4><ul><li>Melakukan supervisi terhadap seluruh aktivitas dan dinamika di departemen HR</li><li>Mengawasi proses rekrutmen, onboarding, dan pengelolaan karyawan</li><li>Mengelola administrasi HR (absensi, kontrak, payroll, dan data karyawan)</li><li>Memastikan kepatuhan terhadap peraturan ketenagakerjaan yang berlaku</li><li>Mengembangkan dan mengimplementasikan kebijakan serta SOP HR</li><li>Menangani hubungan industrial dan permasalahan karyawan</li><li>Melakukan evaluasi kinerja karyawan dan mendukung program pengembangan SDM</li><li>Menyusun laporan HR secara berkala kepada manajemen</li><li>Berkoordinasi dengan departemen lain terkait kebutuhan SDM</li></ul><h4>&nbsp;</h4><h4><strong>Kualifikasi</strong></h4><ul><li>Pendidikan minimal S1 (diutamakan Psikologi, Manajemen SDM, atau terkait)</li><li>Memiliki pengalaman minimal 2–4 tahun di bidang HR (pengalaman sebagai Supervisor menjadi nilai tambah)</li><li>Memahami proses HR secara menyeluruh (rekrutmen, payroll, hubungan industrial, dll.)</li><li>Memiliki kemampuan leadership dan komunikasi yang baik</li><li>Teliti, tegas, dan mampu mengambil keputusan</li><li>Mampu mengoperasikan Microsoft Office</li><li><strong>Mampu berbahasa Mandarin menjadi nilai tambah</strong></li></ul>"}]


# =========================================================
# FUNGSI SCORING
# CATATAN: _edu_level_from_str() sudah dipindah ke Cell 3 agar tersedia saat EDA. Dipanggil ulang di sini hanya sebagai referensi — tidak perlu didefinisikan lagi.
# --- Konstanta ---
PROFICIENCY_WEIGHT = {
    'Kurang'     : 0.25,
    'Cukup'      : 0.50,
    'Baik'       : 0.75,
    'Sangat Baik': 1.00
}

SKILL_THRESHOLD = 0.35
RECENCY_LAMBDA  = 0.15

# Cache embedding
_skill_vec_cache   = {}
_jurusan_vec_cache = {}
_exp_vec_cache     = {}

def get_cached_vec(text, cache_dict):
    if text not in cache_dict:
        cache_dict[text] = model.encode(
            text,
            convert_to_numpy=True
        )
    return cache_dict[text]

print("✓ Konstanta & cache siap")
# ✓ Konstanta & cache siap

# --- S1: Semantic Score ---
def semantic_score(pelamar_vec, lowongan_vec):
    """Cosine similarity profil pelamar vs lowongan."""
    if pelamar_vec is None or lowongan_vec is None:
        return 0.0

    return round(
        float(cos_sim(
            [pelamar_vec],
            [lowongan_vec]
        )[0][0]),
        4
    )


# --- S2: Skill Score ---
def skill_score(
    skills_detail,
    lowongan_vec,
    threshold=SKILL_THRESHOLD
):
    if pd.isna(skills_detail) or not str(skills_detail).strip():
        return 0.0

    if lowongan_vec is None:
        return 0.0

    total_w   = 0.0
    matched_w = 0.0

    for item in str(skills_detail).split(';;'):

        idx = item.rfind('|')
        if idx == -1:
            continue

        nama  = item[:idx].strip()
        level = item[idx+1:].strip()

        w = PROFICIENCY_WEIGHT.get(level, 0.5)
        total_w += w

        skill_text = f"memiliki keahlian {nama}"

        skill_vec = get_cached_vec(
            skill_text,
            _skill_vec_cache
        )

        similarity = float(cos_sim(
            [skill_vec],
            [lowongan_vec]
        )[0][0])

        if similarity >= threshold:
            matched_w += w

    return (
        round(matched_w / total_w, 4)
        if total_w > 0 else 0.0
    )

print("✓ Semantic & Skill scoring siap")
# ✓ Semantic & Skill scoring siap

# --- S3: Education Score ---
def edu_score(
    pendidikan_tertinggi,
    jurusan_tertinggi,
    lowongan_vec
):
    # Dimensi 1: jenjang
    if (
        pd.notna(pendidikan_tertinggi)
        and str(pendidikan_tertinggi).strip()
    ):
        level_int = _edu_level_from_str(
            str(pendidikan_tertinggi)
        )

        level_score = level_int / 10.0
    else:
        level_score = 0.2

    # Dimensi 2: jurusan
    if (
        pd.notna(jurusan_tertinggi)
        and str(jurusan_tertinggi).strip()
        and lowongan_vec is not None
    ):

        jurusan_text = (
            f"latar belakang pendidikan "
            f"jurusan {jurusan_tertinggi}"
        )

        jurusan_vec = get_cached_vec(
            jurusan_text,
            _jurusan_vec_cache
        )

        jurusan_score = float(cos_sim(
            [jurusan_vec],
            [lowongan_vec]
        )[0][0])

    else:
        jurusan_score = 0.3

    return round(
        0.35 * level_score
        + 0.65 * jurusan_score,
        4
    )

print("✓ Education scoring siap")
# ✓ Education scoring siap

# --- S4: Experience Score ---
def exp_score(
    pengalaman_detail,
    lowongan_vec
):

    if (
        pd.isna(pengalaman_detail)
        or not str(pengalaman_detail).strip()
    ):
        return 0.1

    if lowongan_vec is None:
        return 0.1

    scores = []

    pattern = re.compile(
        r'^(.+)\((\d{4})-([\d]+|skrg)\)$'
    )

    for item in str(pengalaman_detail).split(';;'):

        item = item.strip()
        m = pattern.match(item)

        if not m:
            continue

        posisi, thn_awal_str, thn_akhir_str = m.groups()

        thn_awal = int(thn_awal_str)

        thn_akhir = (
            CURRENT_YEAR
            if thn_akhir_str == 'skrg'
            else int(thn_akhir_str)
        )

        thn_akhir = max(
            thn_awal,
            min(thn_akhir, CURRENT_YEAR)
        )

        # relevance
        posisi_text = (
            f"pengalaman kerja "
            f"sebagai {posisi.strip()}"
        )

        posisi_vec = get_cached_vec(
            posisi_text,
            _exp_vec_cache
        )

        relevance = float(cos_sim(
            [posisi_vec],
            [lowongan_vec]
        )[0][0])

        # duration
        durasi = max(
            thn_akhir - thn_awal,
            0
        )

        dur_w = min(
            durasi / 5.0,
            1.0
        )

        # recency
        berlalu = max(
            CURRENT_YEAR - thn_akhir,
            0
        )

        rec_w = math.exp(
            -RECENCY_LAMBDA * berlalu
        )

        score_i = relevance * (
            0.6
            + 0.2 * dur_w
            + 0.2 * rec_w
        )

        scores.append(score_i)

    return (
        round(float(max(scores)), 4)
        if scores else 0.1
    )

print("✓ Experience scoring siap")
# ✓ Experience scoring siap

print("\n--- Test edu level ---")

test_cases = [
    'D4/S1', 'SMA/SMK',
    'S1', 'SMA', 'SMK',
    'D3', 'S2', 'S3',
    'SMP', 'SD', ''
]

for tc in test_cases:
    print(
        f"'{tc}' → "
        f"{_edu_level_from_str(tc)}"
    )


print("\n--- Test regex exp ---")

test_exps = [
    "Team Leader(2018-2026)",
    "Team Leader (Senior)(2018-2026)",
    "Operator produksi(2025-2025)",
    "General Manager HR & GA(2020-skrg)"
]

pattern_test = re.compile(
    r'^(.+)\((\d{4})-([\d]+|skrg)\)$'
)

for t in test_exps:
    m = pattern_test.match(t)

    if m:
        posisi, ta, tb = m.groups()

        print(
            f"OK → "
            f"{posisi.strip()} "
            f"{ta}-{tb}"
        )
    else:
        print(f"FAIL → {t}")

# --- Test edu level ---
# 'D4/S1' → 6
# 'SMA/SMK' → 3
# 'S1' → 6
# 'SMA' → 3
# 'SMK' → 3
# 'D3' → 5
# 'S2' → 8
# 'S3' → 10
# 'SMP' → 2
# 'SD' → 1
# '' → 0

# --- Test regex exp ---
# OK → Team Leader 2018-2026
# OK → Team Leader (Senior) 2018-2026
# OK → Operator produksi 2025-2025
# OK → General Manager HR & GA 2020-skrg

# HITUNG SEMUA SCORE
print("Menghitung semua score komponen...")
print(f"  Total baris : {len(df_all):,}\n")

# Menghitung semua score komponen...
#   Total baris : 5,904

sample_scores = []

for _, row in df_all.sample(20, random_state=42).iterrows():
    lv = lowongan_vec_dict.get(row['idlowongan'])

    if lv is None or pd.isna(row.get('skills_detail')):
        continue

    for item in str(row['skills_detail']).split(';;'):
        idx = item.rfind('|')
        if idx == -1:
            continue

        nama = item[:idx].strip()

        skill_text = f"memiliki keahlian {nama}"
        skill_vec = get_cached_vec(skill_text, _skill_vec_cache)

        sim = float(cos_sim([skill_vec], [lv])[0][0])

        sample_scores.append({
            'skill': nama,
            'sim': round(sim, 3)
        })

pd.DataFrame(sample_scores).sort_values('sim', ascending=False).head(30)

[{"index":25,"skill":"Pelayanan kepada customer","sim":0.694},{"index":8,"skill":"Pemesinan/otomotif/las listrik","sim":0.482},{"index":34,"skill":"Microsoft Office","sim":0.481},{"index":21,"skill":"Bisa mengelola dokumen (arsip), dapat berkomunikasi dengan baik, dan Administrasi dasar","sim":0.474},{"index":23,"skill":"Memahami Instalasi Listrik","sim":0.407},{"index":3,"skill":"Microsoft Office","sim":0.393},{"index":19,"skill":"Microsoft office","sim":0.376},{"index":20,"skill":"Microsoft office","sim":0.376},{"index":4,"skill":"KOMUNIKASI YANG BAIK","sim":0.363},{"index":0,"skill":"Mampu berkomikasi dengan baik","sim":0.349},{"index":26,"skill":"Mengoprasikan mesin washing ampoule , washing vial , mesin tunnel sterilisasi , mesin aoutoclave sterilisasi , mesin las GMAW , robbot wellding dan spot wellding","sim":0.347},{"index":28,"skill":"Instalasi jaringan","sim":0.339},{"index":11,"skill":"Bisa mengoprasikan mesin press,bubut","sim":0.327},{"index":14,"skill":"Mengoprasikan Software Word, Excel & Powerpoint","sim":0.316},{"index":17,"skill":"Pemograman","sim":0.308},{"index":6,"skill":"Microsoft Office (Ms. Excel & Ms. Word)","sim":0.307},{"index":13,"skill":"Creative Digital Desain","sim":0.302},{"index":12,"skill":"microsoft excel, microsoft word, foto, komunikasi, service","sim":0.299},{"index":9,"skill":"DESIGN POSTER","sim":0.298},{"index":24,"skill":"Otomotif","sim":0.296},{"index":7,"skill":"PELAYANAN PRIMA","sim":0.296},{"index":22,"skill":"Programmer","sim":0.292},{"index":1,"skill":"English: Working Knowledge","sim":0.262},{"index":10,"skill":"Microsoft excel, Microsoft word","sim":0.249},{"index":5,"skill":"MENGETIK 10 JARI BUTA","sim":0.231},{"index":29,"skill":"Java script","sim":0.222},{"index":32,"skill":"Phyton","sim":0.213},{"index":31,"skill":"Microsoft Office","sim":0.205},{"index":18,"skill":"Microsoft word","sim":0.199},{"index":2,"skill":"Import/Export Document","sim":0.161}]


s_semantic_list = []
s_skill_list    = []
s_edu_list      = []
s_exp_list      = []

skill_cache_by_pelamar = {}

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

    # skill/edu/exp: get_cached_vec() di dalamnya sudah handle deduplikasi
    # per teks skill/jurusan/posisi → tidak perlu outer cache lagi.
    s_skill_list.append(skill_score(row.get('skills_detail'), lv))
    s_edu_list.append(edu_score(
        row.get('pendidikan_tertinggi'),
        row.get('jurusan_tertinggi'), lv
    ))
    s_exp_list.append(exp_score(row.get('pengalaman_detail'), lv))

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

save_path = '/content/drive/MyDrive/Machine_Learning/TA/scores_computed.csv'

df_all.to_csv(save_path, index=False)

print(f"✓ Tersimpan di: {save_path}")
print("\n  Tersimpan: scores_computed.csv")

# Scoring: 100%|██████████| 5904/5904 [01:43<00:00, 56.82it/s] 

# ✓ Scoring selesai.

# Statistik per komponen (positif vs negatif):
#       semantic_score                                                 skill_score                                            edu_score                                                  exp_score                                             
#                count   mean    std   min    25%    50%    75%    max       count   mean    std  min  25%    50%    75%  max     count   mean    std    min    25%    50%    75%    max     count   mean    std  min  25%    50%    75%    max
# label                                                                                                                                                                                                                                        
# 0             3904.0  0.386  0.102  0.04  0.313  0.388  0.459  0.666      3904.0  0.297  0.405  0.0  0.0  0.000  0.556  1.0    3904.0  0.289  0.089  0.096  0.219  0.270  0.348  0.573    3904.0  0.298  0.155  0.1  0.1  0.329  0.416  0.688
# 1             2000.0  0.413  0.103  0.09  0.340  0.416  0.489  0.689      2000.0  0.376  0.433  0.0  0.0  0.078  1.000  1.0    2000.0  0.298  0.092  0.068  0.225  0.277  0.361  0.614    2000.0  0.307  0.162  0.1  0.1  0.343  0.432  0.697
# ✓ Tersimpan di: /content/drive/MyDrive/Machine_Learning/TA/scores_computed.csv

#   Tersimpan: scores_computed.csv

display(df_all.head())
[{"index":0,"lamaran_id":6,"idpelamar":4,"idlowongan":27,"ideven":"3.0","tanggalmelamar":"18/4/2026","label":1,"namalengkap":"Rifki Muh Sopiandi","jeniskelamin":"Laki-laki","usia":26,"skills_detail":"Exel, Wor, PPT, Sap R3|Sangat Baik","skills_raw":"Exel, Wor, PPT, Sap R3","jumlah_skill":1,"avg_skill_level":"4.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"TKJ","jumlah_pengalaman":"1.0","total_tahun_exp":"8.0","pengalaman_detail":"Team Leader Dept Continues Improvement(2018-2026)","posisi_all":"Team Leader Dept Continues Improvement"},{"index":1,"lamaran_id":7,"idpelamar":4,"idlowongan":25,"ideven":"3.0","tanggalmelamar":"18/4/2026","label":1,"namalengkap":"Rifki Muh Sopiandi","jeniskelamin":"Laki-laki","usia":26,"skills_detail":"Exel, Wor, PPT, Sap R3|Sangat Baik","skills_raw":"Exel, Wor, PPT, Sap R3","jumlah_skill":1,"avg_skill_level":"4.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"TKJ","jumlah_pengalaman":"1.0","total_tahun_exp":"8.0","pengalaman_detail":"Team Leader Dept Continues Improvement(2018-2026)","posisi_all":"Team Leader Dept Continues Improvement"},{"index":2,"lamaran_id":8,"idpelamar":4,"idlowongan":26,"ideven":"3.0","tanggalmelamar":"18/4/2026","label":1,"namalengkap":"Rifki Muh Sopiandi","jeniskelamin":"Laki-laki","usia":26,"skills_detail":"Exel, Wor, PPT, Sap R3|Sangat Baik","skills_raw":"Exel, Wor, PPT, Sap R3","jumlah_skill":1,"avg_skill_level":"4.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"TKJ","jumlah_pengalaman":"1.0","total_tahun_exp":"8.0","pengalaman_detail":"Team Leader Dept Continues Improvement(2018-2026)","posisi_all":"Team Leader Dept Continues Improvement"},{"index":3,"lamaran_id":9,"idpelamar":10,"idlowongan":37,"ideven":"3.0","tanggalmelamar":"20/4/2026","label":1,"namalengkap":"REGY GLASIO C.PRATAMA","jeniskelamin":"Laki-laki","usia":20,"skills_detail":"Mengoperasikan mesin cnc/forklift|Cukup","skills_raw":"Mengoperasikan mesin cnc/forklift","jumlah_skill":1,"avg_skill_level":"2.0","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"Teknik mesin","jumlah_pengalaman":"1.0","total_tahun_exp":"1.0","pengalaman_detail":"Operator produksi(2025-2025)","posisi_all":"Operator produksi"},{"index":4,"lamaran_id":10,"idpelamar":11,"idlowongan":15,"ideven":"3.0","tanggalmelamar":"20/4/2026","label":1,"namalengkap":"M.SAMSUL FIKRI","jeniskelamin":"Laki-laki","usia":18,"skills_detail":"Desain gerfis Corel draw and Canva|Baik;;Editing video cap cut|Cukup;;Excel|Cukup;;Host live|Baik;;Publik spcking|Baik","skills_raw":"Desain gerfis Corel draw and Canva, Editing video cap cut, Excel, Host live, Publik spcking","jumlah_skill":5,"avg_skill_level":"2.6","pendidikan_tertinggi":"SMA/SMK","edu_level_int":3,"jurusan_tertinggi":"Desain Komunikasi Visual","jumlah_pengalaman":"1.0","total_tahun_exp":"3.0","pengalaman_detail":"Magang(2023-2026)","posisi_all":"Magang"}]

# ================================================================================================
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

# --- Separasi per Komponen ---
#   Komponen               Mean Pos   Mean Neg      Delta
#   ----------------------------------------------------
#   semantic_score           0.4130     0.3861    +0.0269
#   skill_score              0.3764     0.2973    +0.0792
#   edu_score                0.2980     0.2888    +0.0093
#   exp_score                0.3071     0.2982    +0.0090

#   Delta positif besar → komponen diskriminatif.


# ===============================================================================================
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

# Total kombinasi bobot: 929
# Menjalankan ablation study...
# Ablation: 100%|██████████| 929/929 [09:19<00:00,  1.66it/s]
# ✓ Ablation selesai.

# Top 10 kombinasi bobot (NDCG@10):
#  w_semantic  w_skill  w_edu  w_exp    ndcg  precision
#        0.50     0.10   0.15   0.25 0.76963    0.26228
#        0.45     0.15   0.15   0.25 0.76870    0.26198
#        0.55     0.05   0.15   0.25 0.76856    0.26228
#        0.45     0.10   0.20   0.25 0.76828    0.26257
#        0.50     0.05   0.25   0.20 0.76827    0.26198
#        0.50     0.10   0.25   0.15 0.76823    0.26169
#        0.50     0.05   0.15   0.30 0.76814    0.26198
#        0.40     0.15   0.20   0.25 0.76803    0.26272
#        0.45     0.10   0.15   0.30 0.76775    0.26302
#        0.60     0.05   0.10   0.25 0.76768    0.26243

# ===============================================================================================
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

# --- Sensitivity Analysis: Top 20 ---
#  w_semantic  w_skill  w_edu  w_exp    ndcg
#        0.50     0.10   0.15   0.25 0.76963
#        0.45     0.15   0.15   0.25 0.76870
#        0.55     0.05   0.15   0.25 0.76856
#        0.45     0.10   0.20   0.25 0.76828
#        0.50     0.05   0.25   0.20 0.76827
#        0.50     0.10   0.25   0.15 0.76823
#        0.50     0.05   0.15   0.30 0.76814
#        0.40     0.15   0.20   0.25 0.76803
#        0.45     0.10   0.15   0.30 0.76775
#        0.60     0.05   0.10   0.25 0.76768
#        0.45     0.15   0.20   0.20 0.76766
#        0.55     0.10   0.15   0.20 0.76759
#        0.55     0.05   0.25   0.15 0.76751
#        0.40     0.05   0.20   0.35 0.76731
#        0.45     0.05   0.25   0.25 0.76721
#        0.55     0.10   0.10   0.25 0.76719
#        0.55     0.10   0.25   0.10 0.76708
#        0.50     0.05   0.20   0.25 0.76707
#        0.55     0.05   0.20   0.20 0.76707
#        0.45     0.05   0.20   0.30 0.76705

#   Std dev bobot di top-20:
#     w_semantic      : 0.0560
#     w_skill         : 0.0373
#     w_edu           : 0.0483
#     w_exp           : 0.0587

#   Std dev rendah → bobot stabil dan reliable.