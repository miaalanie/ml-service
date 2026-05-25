-- ============================================================
-- SQL VIEWS FIXED — JOB MATCHING SYSTEM (v3)
--
-- BUG FIXES dari v2:
--   FIX-1 (v_edu_agg): edu_level_int selalu 0
--          CASE kategori WHEN 'SMA' exact match TIDAK pernah cocok
--          karena DB simpan 'SMA/SMK', 'D4/S1' dll.
--          → Ganti ke CASE WHEN kategori LIKE '%S3%' THEN 10 ...
--
--   FIX-2 (v_edu_agg): pendidikan_tertinggi pakai GROUP_CONCAT
--          dengan SEPARATOR ';;' supaya tidak bentrok dengan koma
--          dalam nilai kategori
--
--   FIX-3 (v_exp_agg): total_tahun_exp bisa negatif jika data korup
--          → Wrap GREATEST(..., 0) di setiap durasi entry
--
--   FIX-4 (v_exp_agg): jurusan_tertinggi pakai SEPARATOR ';;'
--          konsisten dengan pendidikan_tertinggi
-- ============================================================


-- ------------------------------------------------------------
-- VIEW HELPER 1 — Skill Agregasi per Pelamar
-- Tidak ada perubahan dari v2 — sudah benar.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_skill_agg AS
SELECT
    idpelamar,

    -- Format: "namaskill|keterangan;;namaskill|keterangan"
    GROUP_CONCAT(
        CONCAT(namaskill, '|', keterangan)
        ORDER BY namaskill
        SEPARATOR ';;'
    )                                           AS skills_detail,

    -- Format: "namaskill, namaskill, namaskill"
    GROUP_CONCAT(
        namaskill
        ORDER BY namaskill
        SEPARATOR ', '
    )                                           AS skills_raw,

    COUNT(*)                                    AS jumlah_skill,

    -- Skor profisiensi rata-rata (Kurang=1, Cukup=2, Baik=3, SangatBaik=4)
    AVG(CASE keterangan
        WHEN 'Kurang'      THEN 1
        WHEN 'Cukup'       THEN 2
        WHEN 'Baik'        THEN 3
        WHEN 'Sangat Baik' THEN 4
    END)                                        AS avg_skill_level,

    SUM(keterangan = 'Sangat Baik')             AS skill_sangat_baik,
    SUM(keterangan = 'Baik')                    AS skill_baik,
    SUM(keterangan = 'Cukup')                   AS skill_cukup,
    SUM(keterangan = 'Kurang')                  AS skill_kurang

FROM pelamarskills
GROUP BY idpelamar;


-- ------------------------------------------------------------
-- VIEW HELPER 2 — Pendidikan Agregasi per Pelamar
--
-- FIX-1: edu_level_int — pakai LIKE bukan exact match
-- FIX-2: pendidikan_tertinggi — SEPARATOR ';;' agar tidak
--         bentrok dengan koma dalam nilai seperti 'D4/S1'
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_edu_agg AS
SELECT
    idpelamar,

    -- FIX-1: LIKE-based scoring — handle 'SMA/SMK', 'D4/S1', dll.
    -- Order: cek dari level tertinggi dulu agar 'D4/S1' → 6 (S1), bukan error
    MAX(CASE
        WHEN kategori LIKE '%S3%'                   THEN 10
        WHEN kategori LIKE '%S2%'                   THEN 8
        WHEN kategori LIKE '%S1%'                   THEN 6
        WHEN kategori LIKE '%D4%'                   THEN 6
        WHEN kategori LIKE '%D3%'                   THEN 5
        WHEN kategori LIKE '%D2%'                   THEN 4
        WHEN kategori LIKE '%D1%'                   THEN 4
        WHEN kategori LIKE '%SMK%'                  THEN 3
        WHEN kategori LIKE '%SMA%'                  THEN 3
        WHEN kategori LIKE '%SMP%'                  THEN 2
        WHEN kategori LIKE '%SD%'                   THEN 1
        ELSE 0
    END)                                            AS edu_level_int,

    -- FIX-2: SEPARATOR ';;' agar tidak bentrok koma dalam nilai
    -- Mengambil nilai pertama = pendidikan level tertinggi
    SUBSTRING_INDEX(
        GROUP_CONCAT(
            kategori
            ORDER BY CASE
                WHEN kategori LIKE '%S3%'  THEN 10
                WHEN kategori LIKE '%S2%'  THEN 8
                WHEN kategori LIKE '%S1%'  THEN 6
                WHEN kategori LIKE '%D4%'  THEN 6
                WHEN kategori LIKE '%D3%'  THEN 5
                WHEN kategori LIKE '%D2%'  THEN 4
                WHEN kategori LIKE '%D1%'  THEN 4
                WHEN kategori LIKE '%SMK%' THEN 3
                WHEN kategori LIKE '%SMA%' THEN 3
                WHEN kategori LIKE '%SMP%' THEN 2
                WHEN kategori LIKE '%SD%'  THEN 1
                ELSE 0
            END DESC
            SEPARATOR ';;'
        ), ';;', 1
    )                                               AS pendidikan_tertinggi,

    -- Jurusan dari pendidikan tertinggi — FIX-2: SEPARATOR ';;'
    SUBSTRING_INDEX(
        GROUP_CONCAT(
            COALESCE(jurusan, '')
            ORDER BY CASE
                WHEN kategori LIKE '%S3%'  THEN 10
                WHEN kategori LIKE '%S2%'  THEN 8
                WHEN kategori LIKE '%S1%'  THEN 6
                WHEN kategori LIKE '%D4%'  THEN 6
                WHEN kategori LIKE '%D3%'  THEN 5
                WHEN kategori LIKE '%D2%'  THEN 4
                WHEN kategori LIKE '%D1%'  THEN 4
                WHEN kategori LIKE '%SMK%' THEN 3
                WHEN kategori LIKE '%SMA%' THEN 3
                WHEN kategori LIKE '%SMP%' THEN 2
                WHEN kategori LIKE '%SD%'  THEN 1
                ELSE 0
            END DESC
            SEPARATOR ';;'
        ), ';;', 1
    )                                               AS jurusan_tertinggi,

    GROUP_CONCAT(
        DISTINCT jurusan
        ORDER BY jurusan
        SEPARATOR ', '
    )                                               AS jurusan_all,

    COUNT(*)                                        AS jumlah_riwayat_pendidikan

FROM pelamarpendidikans
GROUP BY idpelamar;


-- ------------------------------------------------------------
-- VIEW HELPER 3 — Pengalaman Kerja Agregasi per Pelamar
--
-- FIX-3: total_tahun_exp — GREATEST(..., 0) agar tidak negatif
--         handle kasus "Magang(2025-2025)" = 0 tahun
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_exp_agg AS
SELECT
    idpelamar,

    COUNT(*)                                    AS jumlah_pengalaman,

    -- Format: "posisi(tahunAwal-tahunAkhir);;posisi(tahunAwal-skrg)"
    GROUP_CONCAT(
        CONCAT(
            posisi,
            '(',
            tahunawal,
            '-',
            IFNULL(tahunselesai, 'skrg'),
            ')'
        )
        ORDER BY tahunawal DESC
        SEPARATOR ';;'
    )                                           AS pengalaman_detail,

    GROUP_CONCAT(
        posisi
        ORDER BY tahunawal DESC
        SEPARATOR ', '
    )                                           AS posisi_all,

    SUBSTRING_INDEX(
        GROUP_CONCAT(
            posisi
            ORDER BY tahunawal DESC
        ), ',', 1
    )                                           AS posisi_terakhir,

    -- FIX-3: GREATEST(..., 0) — durasi tidak boleh negatif
    -- Kasus: data korup atau magang singkat (2025-2025 = 0 tahun, valid)
    -- Total tahun pengalaman (aman dari nilai negatif)
    SUM(
        CASE
            WHEN aktif = 1 THEN
                GREATEST(
                    YEAR(CURDATE()) - CAST(tahunawal AS SIGNED),
                    0
                )

            ELSE
                GREATEST(
                    CAST(COALESCE(tahunselesai, YEAR(CURDATE())) AS SIGNED)
                    - CAST(tahunawal AS SIGNED),
                    0
                )
        END
    ) AS total_tahun_exp,
    MAX(aktif)                                  AS sedang_bekerja

FROM pelamarpengalamen
GROUP BY idpelamar;


-- ============================================================
-- VIEW FINAL — Dataset Matching (untuk export ke CSV)
-- ============================================================
CREATE OR REPLACE VIEW v_dataset_matching AS
SELECT
    l.id                                        AS lamaran_id,
    l.idpelamar,
    l.idlowongan,
    l.ideven,
    l.tanggalmelamar,
    1                                           AS label,

    p.namalengkap,
    p.jeniskelamin,
    TIMESTAMPDIFF(YEAR, p.tanggallahir, CURDATE()) AS usia,

    sk.skills_detail,
    sk.skills_raw,
    sk.jumlah_skill,
    sk.avg_skill_level,
    sk.skill_sangat_baik,
    sk.skill_baik,
    sk.skill_cukup,
    sk.skill_kurang,

    ed.pendidikan_tertinggi,
    ed.edu_level_int,
    ed.jurusan_tertinggi,
    ed.jurusan_all,
    ed.jumlah_riwayat_pendidikan,

    ex.jumlah_pengalaman,
    ex.total_tahun_exp,
    ex.pengalaman_detail,
    ex.posisi_all,
    ex.posisi_terakhir,
    ex.sedang_bekerja,

    lo.namalowongan,
    lo.deskripsi                                AS deskripsi_loker,
    lo.kategorilokasi,
    lo.gaji_awal,
    lo.gaji_akhir,
    lo.kuota,
    lo.status                                   AS status_loker,

    kl.id                                       AS idkategori_loker,
    kl.nama                                     AS kategori_loker

FROM lamarans l
JOIN pelamars            p  ON p.id  = l.idpelamar
JOIN lowongans           lo ON lo.id = l.idlowongan
JOIN kategorilowongans   kl ON kl.id = lo.idkategorilowongan
LEFT JOIN v_skill_agg    sk ON sk.idpelamar = l.idpelamar
LEFT JOIN v_edu_agg      ed ON ed.idpelamar = l.idpelamar
LEFT JOIN v_exp_agg      ex ON ex.idpelamar = l.idpelamar;


-- ============================================================
-- QUERY EXPORT CSV
-- ============================================================

-- Export 1: dataset_matching.csv
CREATE OR REPLACE VIEW v_dataset_matching_csv AS
SELECT
    l.id                                        AS lamaran_id,
    l.idpelamar,
    l.idlowongan,
    l.ideven,
    l.tanggalmelamar,
    1                                           AS label,

    p.namalengkap,
    p.jeniskelamin,
    TIMESTAMPDIFF(YEAR, p.tanggallahir, CURDATE()) AS usia,

    sk.skills_detail,
    sk.skills_raw,
    sk.jumlah_skill,
    sk.avg_skill_level,

    ed.pendidikan_tertinggi,
    ed.edu_level_int,
    ed.jurusan_tertinggi,

    ex.jumlah_pengalaman,
    ex.total_tahun_exp,
    ex.pengalaman_detail,
    ex.posisi_all,
    ex.sedang_bekerja,

    lo.namalowongan,
    lo.deskripsi                                AS deskripsi_loker,
    lo.kategorilokasi,
    lo.gaji_awal,
    lo.gaji_akhir,

    kl.id                                       AS idkategori_loker,
    kl.nama                                     AS kategori_loker

FROM lamarans l
JOIN pelamars            p  ON p.id  = l.idpelamar
JOIN lowongans           lo ON lo.id = l.idlowongan
JOIN kategorilowongans   kl ON kl.id = lo.idkategorilowongan
LEFT JOIN v_skill_agg    sk ON sk.idpelamar = l.idpelamar
LEFT JOIN v_edu_agg      ed ON ed.idpelamar = l.idpelamar
LEFT JOIN v_exp_agg      ex ON ex.idpelamar = l.idpelamar;


-- Export 2: all_lowongans.csv
CREATE OR REPLACE VIEW v_lowongans_csv AS
SELECT
    lo.id                                       AS idlowongan,
    lo.namalowongan,
    lo.deskripsi                                AS deskripsi_loker,
    lo.kategorilokasi,
    lo.gaji_awal,
    lo.gaji_akhir,
    lo.status                                   AS status_loker,
    lo.kuota,
    kl.id                                       AS idkategori_loker,
    kl.nama                                     AS kategori_loker
FROM lowongans lo
JOIN kategorilowongans kl ON kl.id = lo.idkategorilowongan;


-- Export 3: all_pelamars.csv
CREATE OR REPLACE VIEW v_pelamars_csv AS
SELECT
    p.id                                        AS idpelamar,
    p.namalengkap,
    p.jeniskelamin,
    TIMESTAMPDIFF(YEAR, p.tanggallahir, CURDATE()) AS usia,

    sk.skills_detail,
    sk.skills_raw,
    sk.jumlah_skill,
    sk.avg_skill_level,

    ed.pendidikan_tertinggi,
    ed.edu_level_int,
    ed.jurusan_tertinggi,

    ex.jumlah_pengalaman,
    ex.total_tahun_exp,
    ex.pengalaman_detail,
    ex.posisi_all,
    ex.sedang_bekerja

FROM pelamars p
LEFT JOIN v_skill_agg    sk ON sk.idpelamar = p.id
LEFT JOIN v_edu_agg      ed ON ed.idpelamar = p.id
LEFT JOIN v_exp_agg      ex ON ex.idpelamar = p.id;
