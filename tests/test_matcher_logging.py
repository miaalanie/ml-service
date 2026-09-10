import logging
import unittest

from app.matcher import MatcherService


def make_vector(value=1.0):
    return [float(value)] * 384


class DummySkill:
    def __init__(self, id_, namaskill, keterangan, embedding=None, nama=None):
        self.id = id_
        self.namaskill = namaskill
        self.nama = nama or namaskill
        self.keterangan = keterangan
        self.embedding = embedding or make_vector()


class DummyEducation:
    def __init__(self, kategori, jurusan='Teknik Informatika'):
        self.kategori = kategori
        self.jurusan = jurusan
        self.embedding = make_vector()


class DummyExperience:
    def __init__(self, posisi='Backend Developer'):
        self.posisi = posisi
        self.bulanawal = 1
        self.tahunawal = 2022
        self.bulanselesai = 6
        self.tahunselesai = 2023
        self.aktif = 0
        self.embedding = make_vector()


class DummySimple:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MatcherLoggingTest(unittest.TestCase):
    def test_match_logs_s1_to_s4_and_reasoning(self):
        pelamar = DummySimple(
            id=11,
            namalengkap='Budi Santoso',
            deskripsidiri='Saya ahli backend dan API.',
            tanggallahir='1997-05-10',
            jeniskelamin='Laki-laki',
            skills=[
                DummySkill(1, 'Python', 'Baik'),
                DummySkill(2, 'FastAPI', 'Baik'),
            ],
            pendidikans=[DummyEducation('S1', 'Teknik Informatika')],
            pengalamans=[DummyExperience('Backend Developer')],
            total_pengalaman_bulan=18,
            embedding=make_vector(),
        )

        lowongan = DummySimple(
            id=21,
            namalowongan='Backend Developer',
            deskripsi='Membuat API dan backend untuk aplikasi.',
            kategori=DummySimple(nama='Teknologi Informasi'),
            kategorilokasi='Sukabumi',
            gaji_awal=5000000,
            gaji_akhir=7000000,
            minimal_pendidikan=DummySimple(kode=7, nama='D4/S1'),
            minimal_pengalaman_bulan=12,
            preferensi_gender='Semua',
            usia_min=18,
            usia_max=40,
            skills=[
                DummySkill(11, 'Python', 'Baik', make_vector(), nama='Python'),
                DummySkill(12, 'FastAPI', 'Baik', make_vector(), nama='FastAPI'),
                DummySkill(13, 'API', 'Cukup', make_vector(), nama='API'),
            ],
            jurusans=[],
            perusahaan_nama='PT Inovasi',
            perusahaan_logo='',
            embedding=make_vector(),
            title_embedding=make_vector(),
        )

        scoring_config = DummySimple(
            semantic_weight=0.25,
            skill_weight=0.25,
            education_weight=0.25,
            experience_weight=0.25,
            skill_threshold=0.50,
            model_dump=lambda: {
                'semantic_weight': 0.25,
                'skill_weight': 0.25,
                'education_weight': 0.25,
                'experience_weight': 0.25,
                'skill_threshold': 0.50,
            },
        )

        payload = DummySimple(
            pelamar=pelamar,
            lowongans=[lowongan],
            scoring_config=scoring_config,
        )

        with self.assertLogs('ml-ranking', level='INFO') as captured:
            MatcherService(embedding_service=object()).match(payload)

        joined = '\n'.join(captured.output)
        self.assertIn('S1', joined)
        self.assertIn('S2', joined)
        self.assertIn('S3', joined)
        self.assertIn('S4', joined)
        self.assertIn('BIODATA', joined)
        self.assertIn('REASONING', joined)
        self.assertIn('Budi Santoso', joined)
        self.assertIn('Backend Developer', joined)


if __name__ == '__main__':
    unittest.main()
