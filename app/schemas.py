from pydantic import BaseModel
from typing import List, Optional


# ============================================================
# SKILL — sesuai tabel pelamarskills
# field: namaskill, keterangan (enum)
# ============================================================
class SkillSchema(BaseModel):
    namaskill: str
    keterangan: str  # 'Kurang' | 'Cukup' | 'Baik' | 'Sangat Baik'


# ============================================================
# PENDIDIKAN — sesuai tabel pelamarpendidikans
# field: kategori, jurusan, tahunawal, tahunselesai
# ============================================================
class PendidikanSchema(BaseModel):
    kategori: str           # 'SMA/SMK' | 'D3' | 'D4/S1' | 'S2' dst
    jurusan: Optional[str] = None
    tahunawal: Optional[int] = None
    tahunselesai: Optional[int] = None


# ============================================================
# PENGALAMAN — sesuai tabel pelamarpengalamen
# field: posisi, tahunawal, tahunselesai, aktif
# TIDAK ada field deskripsi → matching hanya dari posisi
# ============================================================
class PengalamanSchema(BaseModel):
    posisi: str
    tahunawal: int
    tahunselesai: Optional[int] = None
    aktif: int = 0          # 1 = masih bekerja


# ============================================================
# PELAMAR — sesuai tabel pelamars + relasi
# ============================================================
class PelamarSchema(BaseModel):
    id: int
    namalengkap: str
    deskripsidiri: Optional[str] = None   # sering kosong di data real
    skills: List[SkillSchema] = []
    pendidikans: List[PendidikanSchema] = []
    pengalamans: List[PengalamanSchema] = []


# ============================================================
# KATEGORI LOKER — sesuai tabel kategorilowongans
# ============================================================
class KategoriSchema(BaseModel):
    id: int
    nama: str


# ============================================================
# LOWONGAN — sesuai tabel lowongans
# Hanya loker dari event yang sedang aktif yang dikirim
# Filter dilakukan di Laravel sebelum kirim ke ML service
# ============================================================
class LowonganSchema(BaseModel):
    id: int
    namalowongan: str
    deskripsi: str          # HTML dari rich text editor
    kategori: KategoriSchema
    kategorilokasi: Optional[str] = None
    gaji_awal: Optional[float] = None
    gaji_akhir: Optional[float] = None


# ============================================================
# PAYLOAD REQUEST
# Laravel mengirim:
#   - data pelamar lengkap (skills, edu, exp)
#   - list loker aktif dari event yang sedang berjalan
# ============================================================
class MatchRequestSchema(BaseModel):
    pelamar: PelamarSchema
    lowongans: List[LowonganSchema]
