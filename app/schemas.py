from pydantic import BaseModel
from typing import List, Optional


class Skill(BaseModel):
    id: int
    namaskill: str
    keterangan: Optional[str] = None


class Pendidikan(BaseModel):
    id: int
    kategori: str
    namasekolah: str
    jurusan: Optional[str] = None
    tahunawal: Optional[str] = None
    tahunselesai: Optional[str] = None


class Pengalaman(BaseModel):
    id: int
    perusahaan: Optional[str] = None
    jabatan: Optional[str] = None
    mulai: Optional[str] = None
    selesai: Optional[str] = None
    deskripsi: Optional[str] = None


class Pelamar(BaseModel):
    id: int
    namalengkap: str
    deskripsidiri: Optional[str] = None

    pendidikans: List[Pendidikan] = []
    pengalamans: List[Pengalaman] = []
    skills: List[Skill] = []


class Kategori(BaseModel):
    id: int
    nama: str


class Lowongan(BaseModel):
    id: int
    namalowongan: str
    deskripsi: str
    kategori: Kategori


class MatchRequest(BaseModel):
    pelamar: Pelamar
    lowongans: List[Lowongan]