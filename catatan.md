# SLIDE 1 — JUDUL

Assalamu'alaikum warahmatullahi wabarakatuh.

Selamat pagi Bapak/Ibu dosen penguji dan dosen pembimbing.

Perkenalkan, saya **Mia Adelia** dengan NIM **312023004** dari Program Studi Teknik Komputer Politeknik Sukabumi.

Pada kesempatan kali ini saya akan mempresentasikan proposal **Tugas Akhir** yang berjudul **"Pengembangan Sistem Rekomendasi Lowongan dan Pelamar Menggunakan Content-Based Filtering dan Collaborative Filtering, Studi Kasus Dinas Ketenagakerjaan Kota Sukabumi."**

---

# SLIDE 2 — DAFTAR ISI

Pada presentasi ini, pembahasan saya bagi menjadi empat bagian, yaitu **pendahuluan, perancangan sistem, simulasi metode,** dan yang terakhir **kesimpulan.**

---

# PENDAHULUAN — LATAR BELAKANG

Tugas Akhir ini mengangkat studi kasus pada Platform Job Fair Digital milik Dinas Ketenagakerjaan Kota Sukabumi.

Latar belakangnya berawal dari kondisi Kota Sukabumi yang sedang mengalami bonus demografi, sehingga jumlah angkatan kerja terus meningkat. Kondisi tersebut membuat proses rekrutmen perlu dilakukan secara lebih efektif.

Sebagai salah satu upaya, Disnaker menyediakan Platform Job Fair Digital yang mempertemukan perusahaan dengan pencari kerja.

Berbeda dengan platform lowongan kerja pada umumnya, proses rekrutmen di platform ini berbasis event. Artinya, perusahaan perlu terdaftar sebagai peserta event sebelum dapat mempublikasikan lowongan, dan seluruh proses rekrutmen berlangsung dalam periode event tersebut.

Karena seluruh aktivitas rekrutmen berlangsung dalam waktu yang terbatas, proses pencocokan antara pelamar dan lowongan menjadi sangat penting.

Oleh karena itu, pada Tugas Akhir ini saya mengusulkan pengembangan sistem rekomendasi agar pelamar memperoleh lowongan yang lebih sesuai, dan perusahaan juga memperoleh kandidat yang lebih sesuai dengan kebutuhannya.

---

# RUMUSAN MASALAH

Berdasarkan latar belakang tersebut, terdapat empat rumusan masalah yang akan dibahas pada Tugas Akhir ini.

Yang pertama, bagaimana mengembangkan rekomendasi lowongan bagi pelamar.

Kedua, bagaimana mengembangkan rekomendasi kandidat bagi perusahaan.

Ketiga, bagaimana mengimplementasikan metode Content-Based Filtering dan Collaborative Filtering pada Platform Job Fair Digital.

Dan yang terakhir, bagaimana melakukan pengujian terhadap kedua metode tersebut.

---

# BATASAN MASALAH

Agar pembahasan lebih terarah, Tugas Akhir ini memiliki beberapa batasan.

Studi kasus dilakukan pada Platform Job Fair Digital Dinas Ketenagakerjaan Kota Sukabumi.

Rekomendasi yang dikembangkan meliputi rekomendasi lowongan bagi pelamar dan rekomendasi kandidat bagi perusahaan.

Metode yang digunakan adalah **Content-Based Filtering** dan **Collaborative Filtering** yang diterapkan secara terpisah atau **non-hybrid**.

Sedangkan data yang digunakan merupakan **data historis pada platform**, yang terdiri dari **data profil pelamar, data lowongan, data lamaran, dan data wishlist.**

---

# TUJUAN DAN MANFAAT

Tujuan dari Tugas Akhir ini adalah mengembangkan sistem rekomendasi, mengimplementasikan kedua metode tersebut pada platform, serta melakukan pengujian untuk memvalidasi hasil rekomendasi yang dihasilkan.

Adapun manfaatnya yaitu membantu pelamar memperoleh rekomendasi lowongan yang lebih relevan dengan profil yang dimiliki, sekaligus membantu perusahaan memperoleh kandidat yang lebih sesuai sehingga proses seleksi dapat dilakukan dengan lebih efektif.

---

# PERANCANGAN SISTEM — GAMBARAN UMUM

Selanjutnya masuk ke bagian **perancangan sistem**.

Berdasarkan gambaran umum sistem, terdapat tiga aktor utama yaitu **Admin Disnaker, Perusahaan,** dan **Pelamar**.

Seluruh aktivitas pengguna dikelola melalui Platform Job Fair yang dibangun menggunakan **Laravel**.

Kemudian sistem rekomendasi terdiri dari dua metode.

**Content-Based Filtering** dijalankan menggunakan **FastAPI** untuk menghitung tingkat kesesuaian antara pelamar dan lowongan, sekaligus menghasilkan rekomendasi kandidat bagi perusahaan.

Sedangkan **Collaborative Filtering** dijalankan di Laravel untuk menghasilkan rekomendasi berdasarkan riwayat interaksi pengguna pada platform.

Hasil dari kedua metode tersebut kemudian ditampilkan kembali kepada pengguna melalui Platform Job Fair.

---

# ANALISIS DATA

Pada Tugas Akhir ini terdapat tiga kelompok data yang digunakan.

Yang pertama adalah **data pelamar**, yang berisi informasi seperti deskripsi diri, keterampilan, pendidikan, dan pengalaman.

Yang kedua adalah **data lowongan**, yang berisi informasi mengenai kebutuhan perusahaan.

Sedangkan yang ketiga adalah **data interaksi**, yaitu riwayat aktivitas pelamar berupa lamaran dan wishlist.

Data pelamar dan data lowongan digunakan pada metode **Content-Based Filtering**, sedangkan data interaksi digunakan pada metode **Collaborative Filtering**.

---

# METODE 1 — CONTENT-BASED FILTERING

Metode pertama yang digunakan adalah **Content-Based Filtering**.

Metode ini memberikan rekomendasi dengan membandingkan karakteristik pelamar dengan karakteristik lowongan.

Pada Tugas Akhir ini, metode tersebut digunakan dalam dua arah, yaitu memberikan rekomendasi lowongan kepada pelamar, serta memberikan rekomendasi kandidat kepada perusahaan.

Penilaian dilakukan berdasarkan empat komponen, yaitu **semantic score, skill score, education score,** dan **experience score**.

Keempat komponen tersebut kemudian digabungkan menjadi satu nilai akhir atau **final score** sebagai dasar dalam menentukan rekomendasi.

---

# FLOW CONTENT-BASED FILTERING

Secara umum, proses Content-Based Filtering dimulai dari data pelamar dan data lowongan.

Selanjutnya data diproses menjadi representasi teks sehingga dapat dihitung tingkat kemiripannya.

Setelah itu sistem menghitung empat komponen penilaian, yaitu kesamaan isi, kesesuaian keterampilan, pendidikan, dan pengalaman.

Terakhir, seluruh komponen digabungkan menggunakan **weighted scoring** sehingga menghasilkan **final score** yang menjadi dasar pemberian rekomendasi.

---

# METODE 2 — COLLABORATIVE FILTERING

Metode kedua adalah **Collaborative Filtering**.

Berbeda dengan metode sebelumnya, metode ini tidak membandingkan isi profil pelamar maupun isi lowongan.

Collaborative Filtering memanfaatkan riwayat interaksi pengguna pada platform, yaitu data lamaran dan wishlist.

Jika terdapat pelamar lain yang memiliki pola aktivitas yang serupa, maka lowongan yang pernah diminati oleh pelamar tersebut dapat direkomendasikan kepada pelamar aktif, selama lowongan tersebut belum pernah dilamar ataupun disimpan sebelumnya.

---

# FLOW COLLABORATIVE FILTERING

Secara umum prosesnya dimulai ketika pelamar meminta rekomendasi lowongan.

Sistem terlebih dahulu mengambil data yang diperlukan, kemudian mengumpulkan riwayat interaksi seluruh pelamar.

Selanjutnya sistem mencari pelamar yang memiliki pola aktivitas paling mirip dengan pelamar aktif.

Berdasarkan kemiripan tersebut, sistem menghasilkan rekomendasi lowongan yang belum pernah diinteraksi oleh pelamar aktif.

---

# SIMULASI CONTENT-BASED FILTERING

Selanjutnya adalah contoh simulasi menggunakan metode **Content-Based Filtering**.

Pada contoh ini dilakukan pencocokan antara profil seorang pelamar dengan lowongan **Backend Developer**.

Sistem kemudian menghitung empat komponen penilaian.

Hasilnya menunjukkan bahwa deskripsi diri, keterampilan, dan pendidikan sudah sesuai dengan kebutuhan lowongan, sedangkan pengalaman kerja masih belum sepenuhnya memenuhi persyaratan.

Seluruh komponen tersebut kemudian digabungkan menggunakan **weighted scoring** sehingga menghasilkan **final score**.

Karena nilai akhirnya cukup tinggi, maka lowongan tersebut direkomendasikan kepada pelamar.

---

# SIMULASI COLLABORATIVE FILTERING

Pada simulasi ini, pelamar aktif adalah **Pelamar A**.

Sistem kemudian membandingkan riwayat aktivitas Pelamar A dengan pelamar lainnya.

Hasilnya menunjukkan bahwa Pelamar B memiliki histori aktivitas yang paling mirip.

Oleh karena itu, lowongan yang pernah diminati oleh Pelamar B tetapi belum pernah diinteraksi oleh Pelamar A akan memperoleh skor rekomendasi yang lebih tinggi.

---

# KESIMPULAN

Sebagai kesimpulan, Tugas Akhir ini mengusulkan pengembangan sistem rekomendasi pada Platform Job Fair Digital Dinas Ketenagakerjaan Kota Sukabumi.

Sistem memanfaatkan **Content-Based Filtering** untuk menghasilkan rekomendasi berdasarkan tingkat kesesuaian antara pelamar dan lowongan, serta rekomendasi kandidat bagi perusahaan.

Selain itu, sistem juga memanfaatkan **Collaborative Filtering** untuk memperkaya rekomendasi lowongan berdasarkan riwayat interaksi pengguna pada platform.

Selanjutnya, rancangan sistem ini akan diimplementasikan dan diuji sebagai tahap lanjutan dari Tugas Akhir.

---

# SLIDE TERAKHIR

Demikian pemaparan proposal Tugas Akhir yang dapat saya sampaikan.

Terima kasih atas perhatian Bapak dan Ibu dosen.

Saya siap menerima pertanyaan maupun masukan.

---

