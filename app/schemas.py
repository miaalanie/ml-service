from pydantic import BaseModel
from typing import List, Optional


# SKILL — sesuai tabel pelamarskills
class SkillSchema(BaseModel):
    id: Optional[int] = None
    namaskill: str
    keterangan: str  # 'Kurang' | 'Cukup' | 'Baik' | 'Sangat Baik'


# PENDIDIKAN — sesuai tabel pelamarpendidikans
class PendidikanSchema(BaseModel):
    id: Optional[int] = None
    kategori: str
    jurusan: Optional[str] = None
    tahunawal: Optional[int] = None
    tahunselesai: Optional[int] = None


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


# JURUSAN LOWONGAN — dari tabel lowonganjurusans → masterjurusans
class LowonganJurusanSchema(BaseModel):
    id: int
    nama: str


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

class LowonganEmbeddingRequestSchema(BaseModel):
    lowongan: LowonganSchema


# PAYLOAD /match
class MatchRequestSchema(BaseModel):
    pelamar: PelamarSchema
    lowongans: List[LowonganSchema]


class PelamarEmbeddingRequestSchema(BaseModel):
    pelamar: PelamarSchema


# PAYLOAD /rank-applicants
class RankApplicantsRequestSchema(BaseModel):
    lowongan: LowonganSchema
    pelamars: List[PelamarSchema]