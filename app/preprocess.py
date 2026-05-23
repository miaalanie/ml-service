from bs4 import BeautifulSoup
import re


class TextPreprocessor:

    @staticmethod
    def clean_html(text: str):

        if not text:
            return ""

        soup = BeautifulSoup(
            text,
            "html.parser"
        )

        return soup.get_text(separator=" ")


    @staticmethod
    def normalize(text: str):

        text = text.lower()

        text = re.sub(
            r'\s+',
            ' ',
            text
        )

        return text.strip()


    @staticmethod
    def build_pelamar_text(pelamar):

        deskripsi = (
            pelamar.deskripsidiri
            if pelamar.deskripsidiri
            else ""
        )

        skills = " ".join([
            skill.namaskill
            for skill in pelamar.skills
        ])

        pengalaman = " ".join([

            f"""
            {item.jabatan or ''}
            {item.deskripsi or ''}
            """

            for item in pelamar.pengalamans
        ])

        pendidikan = " ".join([

            f"""
            {item.kategori}
            {item.jurusan or ''}
            """

            for item in pelamar.pendidikans
        ])

        text = f"""
        {deskripsi}

        Skills:
        {skills}

        Pengalaman:
        {pengalaman}

        Pendidikan:
        {pendidikan}
        """

        return TextPreprocessor.normalize(text)


    @staticmethod
    def build_lowongan_text(lowongan):

        clean_desc = (
            TextPreprocessor.clean_html(
                lowongan.deskripsi
            )
        )

        text = f"""
        {lowongan.namalowongan}

        Kategori:
        {lowongan.kategori.nama}

        Deskripsi:
        {clean_desc}
        """

        return TextPreprocessor.normalize(text) 