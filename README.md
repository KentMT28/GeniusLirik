Download Model di Drive Berikut: https://drive.google.com/drive/folders/1hRfhAEajImvNzQdnLyIEcP96JhjVNU53?usp=sharing


# GeniusLyrics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)

GeniusLyrics adalah generator lirik lagu otomatis berbasis kecerdasan buatan (AI) yang menggunakan model Bidirectional Long Short-Term Memory (Bi-LSTM) dan embedding Word2Vec. Aplikasi ini menghasilkan lirik koheren dan sesuai genre dalam bahasa Indonesia dan Inggris, berdasarkan dataset dari Genius.com (8.925 lagu Inggris, 3.396 lagu Indonesia). Ideal untuk musisi dan penulis lagu yang mengalami kebuntuan ide (writer's block)!

## Fitur
- **Bahasa Dukungan**: Bahasa Indonesia dan Inggris.
- **Mode Generasi**:
  - Generik: Hasil lirik netral berdasarkan genre.
  - Artis-Spesifik: Meniru gaya artis tertentu (misalnya Taylor Swift untuk Pop).
- **Parameter Kontrol**:
  - Seed text/tema awal.
  - Pilihan genre (Pop, Rock, Rap, Indie, Dangdut).
  - Jumlah kata (50-500).
  - Temperatur (0.5-1.5) untuk mengatur tingkat kreativitas.
- **Evaluasi**: Model dievaluasi dengan validation loss dan perplexity (83.69 untuk Indonesia, 132.88 untuk Inggris).
- **Antarmuka**: Aplikasi web berbasis Flask dengan UI responsif.
- **Filter**: Penyaringan kata kasar untuk output bersih.

## Prasyarat
- Python 3.8 atau lebih tinggi.
- GPU direkomendasikan untuk pelatihan model (opsional untuk inferensi).
- Folder `models` dengan subfolder: `indonesian`, `Bahasa Indonesia Generik V3`, `english_artist`, `english_generic_v4`. (Download model dari [link model jika ada] atau latih sendiri).

## Instalasi
1. Clone repository:
   ```
   git clone https://github.com/username/geniuslyrics.git
   cd geniuslyrics
   ```
2. Install dependensi:
   ```
   pip install -r requirements.txt
   ```
   Catatan: Jika Sastrawi tidak terinstal, jalankan `pip install Sastrawi` untuk preprocessing bahasa Indonesia.

3. Pastikan NLTK data terdownload (otomatis di app.py):
   - punkt
   - stopwords
   - wordnet

4. Siapkan folder `models` dengan file model (.keras, .pkl) sesuai path di `app.py`.

## Cara Penggunaan
1. Jalankan server:
   ```
   python app.py
   ```
   - Akses UI: http://localhost:5000
   - Server berjalan di port 5000, debug mode aktif.

2. Di UI:
   - Pilih tab model (Indonesia Artis, Indonesia Generik, Inggris Generik, Inggris Artis).
   - Isi seed text/tema, genre, artis (jika ada), jumlah kata, dan temperatur.
   - Klik "BUAT LIRIK" atau "GENERATE" untuk menghasilkan lirik.
   - Lirik ditampilkan dengan struktur (Verse, Chorus, Bridge, dll.).

3. API Endpoints (untuk integrasi):
   - `/api/generate` (POST): English Generic.
   - `/api/generate-artist` (POST): English Artist.
   - `/api/generate-indonesian` (POST): Indonesian Artist.
   - `/api/generate-indonesian-generic` (POST): Indonesian Generic.
   - `/api/artists/<genre>` (GET): Daftar artis Inggris per genre.
   - `/api/artists-indonesian/<genre>` (GET): Daftar artis Indonesia per genre.

   Contoh request POST (JSON):
   ```
   {
       "seedText": "rindu di malam sunyi",
       "genre": "pop",
       "temperature": 1.0,
       "num_words": 200
   }
   ```

## Arsitektur
- **Model**: Bi-LSTM dengan 3 layer bertumpuk, dropout 0.2, softmax output.
- **Embedding**: Word2Vec (CBOW) untuk representasi semantik.
- **Preprocessing**: Tokenisasi, stemming (Sastrawi untuk ID), lemmatization (NLTK untuk EN).
- **Dataset**: Web scraping dari Genius.com, diproses untuk genre-spesifik.

## Kontribusi
1. Fork repository.
2. Buat branch: `git checkout -b fitur-baru`.
3. Commit: `git commit -m "Tambah fitur X"`.
4. Push: `git push origin fitur-baru`.
5. Buat Pull Request.

## Lisensi
MIT License. Lihat [LICENSE](LICENSE) untuk detail.
- Dibuat untuk skripsi Universitas Tarumanagara, 2026.

Terima kasih telah menggunakan GeniusLyrics! 🎶 🚀
