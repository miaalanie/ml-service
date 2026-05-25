from sklearn.metrics.pairwise import cosine_similarity
from .scoring import SKILL_THRESHOLD, PROFICIENCY_WEIGHT, _edu_level_from_str


class ReasoningService:
    """
    Explainability layer.

    Tujuan: beri penjelasan konkret kenapa suatu loker direkomendasikan.
    Pendekatan: post-hoc explanation dari score yang sudah dihitung.

    Referensi: Explainable AI (XAI) — Zhang & Chen (2018)
    """

    # ============================================================
    # GENERATE TAGS
    # Tag per komponen: skill match, pendidikan, pengalaman
    # ============================================================
    @staticmethod
    def generate_tags(
        pelamar,
        job_vec,
        embedding_service,
        final_score: float
    ) -> list:
        tags = []

        # --- Skill tags ---
        if pelamar.skills:
            matched_skills = []
            unmatched_skills = []

            for skill in pelamar.skills:
                skill_text = f"memiliki keahlian {skill.namaskill}"
                skill_vec  = embedding_service.encode(skill_text)
                sim = float(
                    cosine_similarity([skill_vec], [job_vec])[0][0]
                )

                if sim >= SKILL_THRESHOLD:
                    matched_skills.append(skill.namaskill)
                else:
                    unmatched_skills.append(skill.namaskill)

            if matched_skills:
                for sk in matched_skills[:3]:  # max 3 tag skill
                    tags.append({
                        'type': 'success',
                        'text': f'{sk} sesuai'
                    })
            else:
                tags.append({
                    'type': 'danger',
                    'text': 'Skill tidak sesuai dengan loker'
                })
        else:
            tags.append({
                'type': 'warning',
                'text': 'Belum ada skill terdaftar'
            })

        # --- Pendidikan tag ---
        if pelamar.pendidikans:
            edu_tertinggi = max(
                pelamar.pendidikans,
                key=lambda p: _edu_level_from_str(p.kategori)
            )
            level = _edu_level_from_str(edu_tertinggi.kategori)

            if level >= 5:  # D3 ke atas
                tags.append({
                    'type': 'success',
                    'text': f'Pendidikan {edu_tertinggi.kategori}'
                })
            else:
                tags.append({
                    'type': 'warning',
                    'text': f'Pendidikan {edu_tertinggi.kategori}'
                })
        else:
            tags.append({
                'type': 'warning',
                'text': 'Data pendidikan tidak ada'
            })

        # --- Pengalaman tag ---
        if pelamar.pengalamans:
            total_exp = sum(
                (exp.tahunselesai or 2026) - exp.tahunawal
                for exp in pelamar.pengalamans
            )
            tags.append({
                'type': 'success',
                'text': f'Pengalaman ±{total_exp} tahun'
            })
        else:
            tags.append({
                'type': 'warning',
                'text': 'Belum ada pengalaman kerja'
            })

        return tags

    # ============================================================
    # GENERATE REASONS
    # Penjelasan tekstual kenapa loker ini direkomendasikan
    # ============================================================
    @staticmethod
    def generate_reasons(
        pelamar,
        lowongan,
        job_vec,
        embedding_service,
        scores: dict
    ) -> list:
        reasons = []

        # Reason skill yang cocok
        matched_skills = []
        for skill in pelamar.skills:
            skill_text = f"memiliki keahlian {skill.namaskill}"
            skill_vec  = embedding_service.encode(skill_text)
            sim = float(
                cosine_similarity([skill_vec], [job_vec])[0][0]
            )
            if sim >= SKILL_THRESHOLD:
                matched_skills.append(
                    f"{skill.namaskill} ({skill.keterangan})"
                )

        if matched_skills:
            reasons.append(
                f"Memiliki skill yang relevan: "
                f"{', '.join(matched_skills)}"
            )
        else:
            reasons.append(
                "Tidak ditemukan skill yang langsung cocok "
                "dengan deskripsi lowongan ini"
            )

        # Reason pendidikan
        if pelamar.pendidikans:
            edu_tertinggi = max(
                pelamar.pendidikans,
                key=lambda p: _edu_level_from_str(p.kategori)
            )
            jurusan_str = (
                f" jurusan {edu_tertinggi.jurusan}"
                if edu_tertinggi.jurusan else ""
            )
            reasons.append(
                f"Pendidikan terakhir {edu_tertinggi.kategori}"
                f"{jurusan_str}"
            )

        # Reason pengalaman
        if pelamar.pengalamans:
            posisi_list = [e.posisi for e in pelamar.pengalamans]
            reasons.append(
                f"Memiliki pengalaman sebagai: "
                f"{', '.join(posisi_list[:3])}"
            )
        else:
            reasons.append("Belum memiliki pengalaman kerja")

        # Reason skor semantic
        sem = scores.get('semantic', 0)
        if sem >= 0.5:
            reasons.append(
                f"Profil secara keseluruhan cukup sesuai "
                f"dengan deskripsi lowongan "
                f"(kecocokan semantik {round(sem*100)}%)"
            )

        return reasons
