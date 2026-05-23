from sklearn.metrics.pairwise import cosine_similarity


class ScoringService:

    @staticmethod
    def semantic_similarity(
        cv_vector,
        job_vector
    ):

        score = cosine_similarity(
            [cv_vector],
            [job_vector]
        )[0][0]

        return float(score)


    @staticmethod
    def skill_match(
        pelamar_skills,
        job_text
    ):

        if len(pelamar_skills) == 0:
            return 0

        matched = 0

        for skill in pelamar_skills:

            if skill.lower() in job_text:
                matched += 1

        return matched / len(pelamar_skills)


    @staticmethod
    def education_match(
        pendidikans,
        job_text
    ):

        for pendidikan in pendidikans:

            kategori = (
                pendidikan.kategori.lower()
            )

            if kategori in job_text:
                return 1

        return 0.5


    @staticmethod
    def experience_match(
        pengalamans,
        job_text
    ):

        if len(pengalamans) == 0:
            return 0.3

        matched = 0

        keywords = [
            'developer',
            'backend',
            'laravel',
            'api',
            'sql',
            'accounting'
        ]

        for pengalaman in pengalamans:

            desc = (
                pengalaman.deskripsi or ''
            ).lower()

            for keyword in keywords:

                if (
                    keyword in desc and
                    keyword in job_text
                ):
                    matched += 1
                    break

        return min(
            matched / len(pengalamans),
            1
        )


    @staticmethod
    def final_score(
        semantic_score,
        skill_score,
        edu_score,
        exp_score
    ):

        score = (
            semantic_score * 0.5 +
            skill_score * 0.25 +
            edu_score * 0.1 +
            exp_score * 0.15
        )

        return round(score, 2)


    @staticmethod
    def classify(score):

        if score >= 0.8:
            return 'Sangat Cocok'

        if score >= 0.6:
            return 'Cukup Cocok'

        return 'Kurang Cocok'