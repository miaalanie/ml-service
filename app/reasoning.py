class ReasoningService:

    @staticmethod
    def generate_tags(
        pelamar,
        lowongan_text,
        final_score
    ):

        tags = []

        pelamar_skills = [
            skill.namaskill.lower()
            for skill in pelamar.skills
        ]

        for skill in pelamar_skills:

            if skill in lowongan_text:

                tags.append({
                    'type': 'success',
                    'text': f'{skill.upper()} sesuai'
                })

        if len(tags) == 0:

            tags.append({
                'type': 'danger',
                'text': 'Skill tidak sesuai'
            })

        pendidikan_match = False

        for pendidikan in pelamar.pendidikans:

            kategori = (
                pendidikan.kategori.lower()
            )

            if kategori in lowongan_text:

                pendidikan_match = True

                tags.append({
                    'type': 'success',
                    'text': 'Pendidikan sesuai'
                })

                break

        if not pendidikan_match:

            tags.append({
                'type': 'warning',
                'text': 'Pendidikan kurang sesuai'
            })

        if len(pelamar.pengalamans) > 0:

            tags.append({
                'type': 'success',
                'text': 'Pengalaman ada'
            })

        else:

            tags.append({
                'type': 'warning',
                'text': 'Belum ada pengalaman'
            })

        return tags


    @staticmethod
    def generate_reasons(
        pelamar,
        lowongan_text
    ):

        reasons = []

        for skill in pelamar.skills:

            if (
                skill.namaskill.lower()
                in lowongan_text
            ):

                reasons.append(
                    f"Skill {skill.namaskill} cocok dengan lowongan"
                )

        if len(pelamar.pengalamans) > 0:

            reasons.append(
                "Memiliki pengalaman kerja"
            )

        if len(pelamar.pendidikans) > 0:

            reasons.append(
                "Latar belakang pendidikan tersedia"
            )

        return reasons


    @staticmethod
    def determine_color(score):

        if score >= 0.8:
            return 'green'

        if score >= 0.6:
            return 'yellow'

        return 'red'


    @staticmethod
    def percentage(score):

        return round(score * 100)