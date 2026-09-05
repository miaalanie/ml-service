from pydantic import BaseModel, model_validator
from typing import List, Optional


class ScoringConfigSchema(BaseModel):
    semantic_weight: float = 0.25
    skill_weight: float = 0.25
    education_weight: float = 0.25
    experience_weight: float = 0.25
    skill_threshold: float = 0.50

    @model_validator(mode='after')
    def validate_weights(self):
        weights = (
            self.semantic_weight,
            self.skill_weight,
            self.education_weight,
            self.experience_weight,
        )
        if any(weight < 0 or weight > 1 for weight in weights):
            raise ValueError('Bobot harus berada di antara 0 dan 1.')
        if abs(sum(weights) - 1.0) > 0.0001:
            raise ValueError('Total bobot harus sama dengan 1.0.')
        if not 0 <= self.skill_threshold <= 1:
            raise ValueError('Skill threshold harus berada di antara 0 dan 1.')
        return self


# SKILL — sesuai tabel pelamarskills
class SkillSchema(BaseModel):
    id: Optional[int] = None
    namaskill: str
    keterangan: str  # 'Kurang' | 'Cukup' | 'Baik' | 'Sangat Baik'
    embedding: Optional[List[float]] = None


# PENDIDIKAN — sesuai tabel pelamarpendidikans
class PendidikanSchema(BaseModel):
    id: Optional[int] = None
    kategori: str
    jurusan: Optional[str] = None
    tahunawal: Optional[int] = None
    tahunselesai: Optional[int] = None
    embedding: Optional[List[float]] = None


# PENGALAMAN — sesuai tabel pelamarpengalamen
# bulanawal & bulanselesai ditambah sesuai ALTER TABLE
class PengalamanSchema(BaseModel):
    id: Optional[int] = None
    posisi: str
    bulanawal: int = 0      # 0 = tidak diketahui
    tahunawal: int
    bulanselesai: int = 0   # 0 = tidak diketahui
    tahunselesai: Optional[int] = None
    aktif: int = 0          # 1 = masih bekerja
    embedding: Optional[List[float]] = None


# PELAMAR — sesuai tabel pelamars + relasi
# total_pengalaman_bulan dihitung di Laravel (MLMatchingService)
class PelamarSchema(BaseModel):
    id: int
    namalengkap: str
    deskripsidiri: Optional[str] = None
    tanggallahir: Optional[str] = None
    jeniskelamin: Optional[str] = None
    skills: List[SkillSchema] = []
    pendidikans: List[PendidikanSchema] = []
    pengalamans: List[PengalamanSchema] = []
    total_pengalaman_bulan: int = 0
    embedding: Optional[List[float]] = None


# KATEGORI LOKER — sesuai tabel kategorilowongans
class KategoriSchema(BaseModel):
    id: int
    nama: str


# MINIMAL PENDIDIKAN — dari kolom minimal_pendidikan (TINYINT)
# kode: 1=SD, 2=SMP, 3=SMA/SMK, 4=D1, 5=D2, 6=D3, 7=D4/S1, 8=S2, 9=S3
class MinimalPendidikanSchema(BaseModel):
    kode: int
    nama: str


# SKILL LOWONGAN — dari tabel lowonganskills → masterskills
class LowonganSkillSchema(BaseModel):
    id: int
    nama: str
    embedding: Optional[List[float]] = None


# JURUSAN LOWONGAN — dari tabel lowonganjurusans → masterjurusans
class LowonganJurusanSchema(BaseModel):
    id: int
    nama: str
    embedding: Optional[List[float]] = None


# LOWONGAN — sesuai tabel lowongans + kolom tambahan
class LowonganSchema(BaseModel):
    id: int
    namalowongan: str
    deskripsi: str
    kategori: KategoriSchema
    kategorilokasi: Optional[str] = None
    gaji_awal: Optional[float] = None
    gaji_akhir: Optional[float] = None

    # Kolom tambahan dari ALTER TABLE
    minimal_pendidikan: Optional[MinimalPendidikanSchema] = None
    minimal_pengalaman_bulan: int = 0
    preferensi_gender: str = 'Semua'    # 'Semua' | 'Laki-laki' | 'Perempuan'
    usia_min: int = 0
    usia_max: int = 0

    # Relasi skills & jurusans yang diharapkan loker
    skills: List[LowonganSkillSchema] = []
    jurusans: List[LowonganJurusanSchema] = []

    perusahaan_nama: Optional[str] = None
    perusahaan_logo: Optional[str] = None
    embedding: Optional[List[float]] = None
    title_embedding: Optional[List[float]] = None

class LowonganEmbeddingRequestSchema(BaseModel):
    lowongan: LowonganSchema


# PAYLOAD /match
class MatchRequestSchema(BaseModel):
    pelamar: PelamarSchema
    lowongans: List[LowonganSchema]
    scoring_config: ScoringConfigSchema = ScoringConfigSchema()


class PelamarEmbeddingRequestSchema(BaseModel):
    pelamar: PelamarSchema


# PAYLOAD /rank-applicants
class RankApplicantsRequestSchema(BaseModel):
    lowongan: LowonganSchema
    pelamars: List[PelamarSchema]
    scoring_config: ScoringConfigSchema = ScoringConfigSchema()