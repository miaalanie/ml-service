import re
import unicodedata
from bs4 import BeautifulSoup


# Mapping level pendidikan — dipakai S3 (Education Score)
# Skala 1-9 sesuai definisi scoring
PENDIDIKAN_LEVEL = {
    'SD' : 1,
    'SMP': 2,
    'SMA': 3, 'SMK': 3,
    'D1' : 4,
    'D2' : 5,
    'D3' : 6,
    'D4' : 7, 'S1': 7,
    'S2' : 8,
    'S3' : 9,
}

# Mapping kode DB (minimal_pendidikan TINYINT) ke level yang sama
# 1=SD,2=SMP,3=SMA/SMK,4=D1,5=D2,6=D3,7=D4/S1,8=S2,9=S3
KODE_TO_LEVEL = {
    1: 1,   # SD
    2: 2,   # SMP
    3: 3,   # SMA/SMK
    4: 4,   # D1
    5: 5,   # D2
    6: 6,   # D3
    7: 7,   # D4/S1
    8: 8,   # S2
    9: 9,   # S3
}

# Mapping keterangan skill ke bobot numerik — dipakai S2 (Skill Score)
SKILL_WEIGHT = {
    'kurang'      : 0.25,
    'cukup'       : 0.50,
    'baik'        : 0.75,
    'sangat baik' : 1.00,
}

SKILL_THRESHOLD = 0.5


class TextPreprocessor:

    # ============================================================
    # STEP 1 — HTML Stripping
    # ============================================================
    @staticmethod
    def clean_html(text: str) -> str:
        if not text:
            return ""
        soup = BeautifulSoup(str(text), "html.parser")
        return soup.get_text(separator=" ")

    # ============================================================
    # STEP 2 — Unicode Normalization
    # ============================================================
    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", str(text))

    # ============================================================
    # STEP 3 — Lowercasing
    # STEP 4 — Whitespace Normalization
    # ============================================================
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = TextPreprocessor.normalize_unicode(text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ============================================================
    # STEP 5 — Structured Representation: Pelamar
    # Dipakai S1 (Semantic Score) — embedding profil pelamar
    # ============================================================
    @staticmethod
    def build_pelamar_text(pelamar) -> str:
        parts = []

        if pelamar.deskripsidiri and pelamar.deskripsidiri.strip():
            parts.append(
                f"profil diri: {pelamar.deskripsidiri}"
            )

        if pelamar.skills:
            skill_names = ", ".join([
                s.namaskill for s in pelamar.skills
            ])
            parts.append(
                f"keahlian yang dimiliki: {skill_names}"
            )

        if pelamar.pendidikans:
            edu = TextPreprocessor._get_pendidikan_tertinggi(
                pelamar.pendidikans
            )
            if edu:
                jurusan = edu.jurusan or ""
                parts.append(
                    f"latar belakang pendidikan {edu.kategori}"
                    + (f" jurusan {jurusan}" if jurusan else "")
                )

        if pelamar.pengalamans:
            posisi_list = ", ".join([
                p.posisi for p in pelamar.pengalamans
            ])
            parts.append(
                f"pengalaman kerja sebagai: {posisi_list}"
            )

        return TextPreprocessor.normalize_text(" ".join(parts))

    # ============================================================
    # STEP 5 — Structured Representation: Lowongan
    # Dipakai S1 (Semantic Score) — embedding teks loker
    # ============================================================
    @staticmethod
    def build_lowongan_text(lowongan) -> str:
        parts = []

        if lowongan.namalowongan:
            parts.append(
                f"posisi yang dibutuhkan: {lowongan.namalowongan}"
            )

        if lowongan.kategori and lowongan.kategori.nama:
            parts.append(
                f"kategori pekerjaan: {lowongan.kategori.nama}"
            )

        if lowongan.deskripsi:
            bersih = TextPreprocessor.clean_html(lowongan.deskripsi)
            if bersih.strip():
                parts.append(
                    f"deskripsi pekerjaan: {bersih}"
                )

        # Skill yang diminta — memperkaya konteks semantik
        if lowongan.skills:
            skill_names = ", ".join([
                s.nama for s in lowongan.skills
            ])
            parts.append(
                f"keahlian yang dibutuhkan: {skill_names}"
            )

        # Jurusan yang diutamakan — memperkaya konteks semantik
        if lowongan.jurusans:
            jurusan_names = ", ".join([
                j.nama for j in lowongan.jurusans
            ])
            parts.append(
                f"jurusan yang diutamakan: {jurusan_names}"
            )

        # Minimal pendidikan — konteks semantik
        if lowongan.minimal_pendidikan:
            parts.append(
                f"minimal pendidikan: {lowongan.minimal_pendidikan.nama}"
            )

        return TextPreprocessor.normalize_text(" ".join(parts))

    # ============================================================
    # HELPER — S2: Skill Score
    # Ambil bobot keterangan skill pelamar
    # ============================================================
    @staticmethod
    def get_skill_weight(keterangan: str) -> float:
        return SKILL_WEIGHT.get(keterangan.lower().strip(), 0.0)

    # ============================================================
    # HELPER — S3: Education Score
    # Ambil level jenjang dari string kategori pendidikan
    # ============================================================
    @staticmethod
    def get_pendidikan_level(kategori: str) -> int:
        k = str(kategori).upper().strip()
        for key, val in PENDIDIKAN_LEVEL.items():
            if key in k:
                return val
        return 0

    # ============================================================
    # HELPER — S3: Education Score
    # Konversi kode DB (minimal_pendidikan.kode) ke level
    # ============================================================
    @staticmethod
    def get_pendidikan_level_from_kode(kode: int) -> int:
        return KODE_TO_LEVEL.get(kode, 0)

    # ============================================================
    # HELPER — S3: Education Score
    # Ambil jurusan pelamar dari pendidikan tertinggi
    # SD/SMP tidak punya jurusan → return None
    # ============================================================
    @staticmethod
    def get_jurusan_pelamar(pendidikans) -> str | None:
        edu = TextPreprocessor._get_pendidikan_tertinggi(pendidikans)
        if not edu:
            return None
        jurusan = (edu.jurusan or "").strip()
        # SD/SMP tidak punya jurusan
        if not jurusan or jurusan == "-":
            return None
        level = TextPreprocessor.get_pendidikan_level(edu.kategori)
        if level <= 2:  # SD=1, SMP=2
            return None
        return jurusan

    # ============================================================
    # HELPER — S4: Experience Score
    # Ambil semua posisi pengalaman pelamar sebagai list string
    # untuk dicari posisi paling cocok dengan loker
    # ============================================================
    @staticmethod
    def get_posisi_pelamar(pengalamans) -> list[str]:
        if not pengalamans:
            return []
        return [
            TextPreprocessor.normalize_text(p.posisi)
            for p in pengalamans
            if p.posisi and p.posisi.strip()
        ]

    # ============================================================
    # INTERNAL — Pendidikan Tertinggi
    # ============================================================
    @staticmethod
    def _get_pendidikan_tertinggi(pendidikans):
        if not pendidikans:
            return None
        return max(
            pendidikans,
            key=lambda p: TextPreprocessor.get_pendidikan_level(p.kategori)
        )