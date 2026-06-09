import re
import unicodedata
from bs4 import BeautifulSoup


class TextPreprocessor:

    # STEP 1 — HTML Stripping
    @staticmethod
    def clean_html(text: str) -> str:
        if not text:
            return ""
        soup = BeautifulSoup(str(text), "html.parser")
        return soup.get_text(separator=" ")


    # STEP 2 — Unicode Normalization
    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", str(text))

    # STEP 3 — Lowercasing
    # STEP 4 — Whitespace Normalization
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = TextPreprocessor.normalize_unicode(text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # STEP 5 — Structured Representation
    @staticmethod
    def build_pelamar_text(pelamar) -> str:
        parts = []

        # Deskripsi diri — opsional, sering kosong
        if pelamar.deskripsidiri and pelamar.deskripsidiri.strip():
            parts.append(
                f"profil diri: {pelamar.deskripsidiri}"
            )

        # Skills — nama skill saja (keterangan untuk scoring terpisah)
        if pelamar.skills:
            skill_names = ", ".join([
                s.namaskill for s in pelamar.skills
            ])
            parts.append(
                f"keahlian yang dimiliki: {skill_names}"
            )

        # Pendidikan tertinggi saja
        if pelamar.pendidikans:
            edu_tertinggi = TextPreprocessor._get_pendidikan_tertinggi(
                pelamar.pendidikans
            )
            if edu_tertinggi:
                jurusan = edu_tertinggi.jurusan or ""
                parts.append(
                    f"latar belakang pendidikan "
                    f"{edu_tertinggi.kategori} "
                    f"jurusan {jurusan}"
                )

        # Pengalaman kerja — hanya posisi (tidak ada deskripsi di DB)
        if pelamar.pengalamans:
            posisi_list = ", ".join([
                p.posisi for p in pelamar.pengalamans
            ])
            parts.append(
                f"pengalaman kerja sebagai: {posisi_list}"
            )

        return TextPreprocessor.normalize_text(" ".join(parts))

    # Build teks lowongan untuk embedding
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

        return TextPreprocessor.normalize_text(" ".join(parts))

    @staticmethod
    def _get_pendidikan_tertinggi(pendidikans):
        if not pendidikans:
            return None

        def level(p):
            k = str(p.kategori).upper()
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

        return max(pendidikans, key=level)
