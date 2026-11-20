# Analisis Sentimen Ulasan Pariwisata Pulau Jawa Menggunakan IndoBERTweet
**Studi Komparatif Strategi Fine-Tuning dan Pengaruh Emoji**

**Peneliti:** Achmal Farhan Ashidiqy (202210370311298) — Program Studi Informatika

---

## Ringkasan Proyek

Proyek ini membangun alur kerja (*pipeline*) *Natural Language Processing* (NLP) secara menyeluruh (*end-to-end*) untuk mengklasifikasikan sentimen ulasan pengunjung objek wisata di Pulau Jawa ke dalam tiga kelas (**Positif**, **Negatif**, dan **Netral**). Karena ulasan pariwisata di Google Maps didominasi oleh bahasa Indonesia informal, singkatan, *slang*, *code-switching* Inggris–Indonesia, penggunaan emoji, serta ketimpangan distribusi kelas (*class imbalance*), penelitian ini mengevaluasi efektivitas arsitektur Transformer **IndoBERTweet** dibandingkan model klasik **TF-IDF + Logistic Regression** melalui **12 skenario eksperimen terkontrol**.

### Pencapaian Utama
* **Model Terbaik (Run 9 — IndoBERTweet `Frozen-6 Layers` + `Dropout 0,1` + `No Emoji`):** Mencapai **Macro F1-Score sebesar 0,7698** dan **Akurasi 86,01%** pada *test set*, meningkat **7,59%** secara relatif dibandingkan *baseline* terbaik TF-IDF + Logistic Regression (Macro F1-Score **0,7155**).
* **Strategi *Partial Fine-Tuning* Optimal:** Membekukan 6 *layer encoder* awal (`layer 0–5`) dan melatih 6 *layer* atas (`layer 6–11`) terbukti menjadi titik seimbang (*sweet spot*) yang mencegah *domain mismatch* sekaligus menekan risiko *overfitting* dan *catastrophic forgetting* pada dataset terbatas.
* **Temuan *The Emoji Effect Paradox*:** Penambahan informasi emoji memberikan peningkatan kecil pada model klasik TF-IDF + Logistic Regression, tetapi justru menurunkan performa pada seluruh konfigurasi IndoBERTweet yang di-*fine-tune* akibat ambiguitas semantik emoji pada ulasan negatif.

---

## Teknologi dan Infrastruktur

* **Bahasa Pemrograman:** Python
* **Lingkungan Komputasi:** Google Colaboratory (GPU Tesla T4, VRAM 15,6 GB)
* **Library & Framework:** PyTorch, HuggingFace Transformers (`indolem/indobertweet-base-uncased`), Sastrawi (`StopWordRemover`), dan modul `Counter` Python
* **Sumber Data:** Google Maps Places API (*endpoint* `nearbysearch` dan `place/details`)

---

## Alur Metodologi dan Pipeline Data

### 1. Pengumpulan Data (*Spatial Grid Scanning*)
* **Pemindaian Geospasial:** Fungsi `discovery_grid()` dan `fetch_all_nearby()` membagi wilayah *bounding box* Pulau Jawa (Lintang $-8,80$ s.d. $-5,50$ dan Bujur $105,0$ s.d. $114,6$) menjadi kisi koordinat dengan langkah $0,45^\circ$ (~50 km), radius pencarian $50.000\text{ m}$, filter tipe `tourist_attraction`, serta *pagination* (`next_page_token`). Tahap ini mengidentifikasi **4.675 titik wisata unik** (`places_stage1.csv`).
* **Ekstraksi & Filtrasi Ulasan:** Fungsi `fetch_reviews()` dan `fetch_place_details()` dengan parameter `reviews_sort=newest` dan `reviews_no_translations=true` (untuk mencegah kontaminasi terjemahan otomatis Google) mengekstrak **15.446 ulasan mentah** berbagai bahasa (`reviews_stage2.csv`). Fungsi `filter_indonesian_reviews_in_java()` kemudian menyaring ulasan berbahasa Indonesia (`lang=="id"`) yang berada di dalam koordinat Pulau Jawa, menghasilkan **6.206 ulasan awal** (`reviews_indonesian_java.csv`).

### 2. Penyusutan Data (*Data Funnel*) dan Pelabelan (*Majority Voting*)
Dataset awal memiliki 6 kolom label *Aspect-Based Sentiment Analysis* (ABSA) dengan tingkat kekosongan (*sparsity*) tinggi (misalnya kolom *Price and Value* kosong sebesar 90,7%). Untuk menyesuaikan dengan kebutuhan klasifikasi tingkat dokumen pada IndoBERTweet, diterapkan tahapan penyaringan dan agregasi label sebagai berikut:

| Tahap | Proses | Jumlah Data | Penyusutan |
| :--- | :--- | :---: | :---: |
| **1** | Dataset hasil filter bahasa Indonesia & wilayah Pulau Jawa | 6.206 baris | — |
| **2** | Filter temporal ulasan tahun 2021–2025 | 5.905 baris | $-301$ ($4,85\%$) |
| **3** | Pelabelan **Majority Voting** (hapus baris dengan nilai `NaN` pada seluruh 6 aspek) | 4.913 baris | $-992$ ($16,80\%$) |
| **4** | Preprocessing teks & validasi panjang minimal 2 kata (**Dataset Final**) | **4.783 baris** | $-130$ ($2,65\%$) |

* **Mekanisme Majority Voting:** Menghitung frekuensi kemunculan sentimen (`positif`, `netral`, `negatif`) dari keenam kolom aspek menggunakan `Counter`, dengan aturan *tie-breaking*: `positif > netral > negatif` (`df_labeled_majority_voting_2021_2025.csv`).
* **Ketimpangan Kelas (*Class Imbalance*):** Pasca-pelabelan, kelas Positif mendominasi sebesar **75,94%** (3.731 sampel), diikuti Negatif **12,31%** (605 sampel), dan Netral **11,74%** (577 sampel) dengan rasio imbalansi **6,47**.

### 3. Preprocessing Teks (6 Tahap Sekuensial)
Pipeline preprocessing dijalankan secara berurutan **tanpa menerapkan *stemming*** agar informasi morfologis tetap terjaga bagi *WordPiece Tokenizer* IndoBERTweet:
1. ***Case Folding* & *Noise Removal*:** Mengubah huruf menjadi huruf kecil serta menghapus tanda baca, URL, tag HTML, dan angka.
2. **Normalisasi Karakter Berulang:** Fungsi `normalize_repeated_chars()` memangkas karakter berulang lebih dari dua kali menjadi satu karakter tunggal (misalnya *"baguuuuss"*).
3. **Normalisasi Kamus Korpus (`NORM_DICT`):** Kamus mandiri yang memetakan kata non-standar pada ulasan wisata mencakup negasi (*"ga"*, *"gak"* $\rightarrow$ *"tidak"*), *intensifier* (*"bgt"* $\rightarrow$ *"banget"*), singkatan (*"krn"* $\rightarrow$ *"karena"*), *slang* (*"mantul"* $\rightarrow$ *"mantap betul"*), dan *code-switching* Inggris–Indonesia (*"worth it"* $\rightarrow$ *"sepadan"*, *"view"* $\rightarrow$ *"pemandangan"*).
4. **Penanganan Emoji (2 Skenario Paralel):**
   * **No Emoji (`df_preprocessed_no_emoji`):** Menghapus seluruh token emoji.
   * **Keep Emoji (`df_preprocessed_keep_emoji`):** Mengonversi emoji menjadi deskripsi teks (misalnya `thumbs_up`).
5. ***Stopword Removal* Kustom:** Menggunakan modul Sastrawi dengan **mengecualikan kata negasi** (*"tidak"*, *"bukan"*, *"jangan"*, *"kurang"*) agar tidak terjadi pembalikan makna sentimen (*negation flipping*).
6. **Validasi Panjang Teks:** Menghapus ulasan yang menyisakan kurang dari 2 kata, menghasilkan **4.783 baris** dataset siap latih (komposisi akhir: Positif 75,96%, Negatif 12,44%, Netral 11,60%).

### 4. Pembagian Dataset dan Penanganan Imbalance
* **Stratified Split (80:10:10):** Menggunakan mekanisme ID Indeks (`X_idx`) agar komposisi baris data identik pada skenario *No Emoji* dan *Keep Emoji*: **3.826 data latih**, **478 data validasi**, dan **479 data uji**.
* **Class Weighting:** Menerapkan pembobotan inversi frekuensi pada fungsi `CrossEntropyLoss` PyTorch (bobot **0,43** untuk Positif, **2,67** untuk Negatif, dan **2,87** untuk Netral) tanpa mengubah komposisi fisik teks.

---

## Arsitektur & Konfigurasi Pelatihan Model

1. **Baseline Model (TF-IDF + Logistic Regression):**
   * `TfidfVectorizer` (`max_features=15000`, `ngram_range=(1,2)`, `analyzer='word'`).
   * `LogisticRegression` (`solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`).
2. **Model Utama (IndoBERTweet — `indolem/indobertweet-base-uncased`):**
   * **Tokenizer:** `max_length=128` dengan *padding* dan *truncation*.
   * **Classifier Head:** Representasi token `[CLS]` berdimensi $768$ diteruskan ke lapisan `Dropout` dan `Linear` ($768 \times 3$), lalu fungsi aktivasi `Softmax` saat inferensi.
   * **Hyperparameter Pelatihan:** Optimizer `AdamW` (*learning rate* $2 \times 10^{-5}$, *weight decay* $0,01$), maksimal 5 *epoch*, *early stopping* (`patience=2` berdasarkan *validation loss*), *linear warmup* ($10\%$ langkah awal) diikuti *linear decay*, serta *gradient clipping* (`max_norm=1.0`).

---

## Hasil Eksperimen (12 Run)

Karena adanya ketimpangan kelas dengan rasio 6,47, **Macro F1-Score** ditetapkan sebagai metrik evaluasi utama agar performa pada kelas minoritas (Negatif dan Netral) dinilai secara proporsional. Pencatatan metrik distandardisasi menggunakan fungsi `evaluate_and_log()` dengan *random seed* yang konsisten.

| Run | Model | Mode Emoji | Strategi Freeze | Dropout | Accuracy (%) | Macro F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | TF-IDF + LR | No Emoji | — | — | 82,25% | 0,7122 |
| **2** | TF-IDF + LR | Keep Emoji | — | — | 82,46% | 0,7155 |
| **3** | IndoBERTweet | No Emoji | Frozen All | Default | 55,53% | 0,3725 |
| **4** | IndoBERTweet | Keep Emoji | Frozen All | Default | 50,94% | 0,3846 |
| **5** | IndoBERTweet | No Emoji | Full Fine-Tuning | Default | 84,97% | 0,7481 |
| **6** | IndoBERTweet | Keep Emoji | Full Fine-Tuning | Default | 82,46% | 0,7294 |
| **7** | IndoBERTweet | No Emoji | Frozen-6 (Layer 0–5) | Default | 85,59% | 0,7687 |
| **8** | IndoBERTweet | Keep Emoji | Frozen-6 (Layer 0–5) | Default | 83,51% | 0,7174 |
| **9** | **IndoBERTweet (Terbaik)** | **No Emoji** | **Frozen-6 (Layer 0–5)** | **0,1** | **86,01%** | **0,7698** |
| **10** | IndoBERTweet | Keep Emoji | Frozen-6 (Layer 0–5) | 0,1 | 82,88% | 0,7303 |
| **11** | IndoBERTweet | No Emoji | Frozen-6 (Layer 0–5) | 0,5 | 84,13% | 0,7342 |
| **12** | IndoBERTweet | Keep Emoji | Frozen-6 (Layer 0–5) | 0,5 | 84,97% | 0,7388 |

*Catatan metrik pendukung Model Terbaik (Run 9): Macro Precision sebesar **0,7488** dan Macro Recall sebesar **0,7965**.*

---

## Analisis Temuan & Error Analysis

### 1. Perbandingan Strategi Fine-Tuning
* **Frozen All (Run 3–4):** Mengalami penurunan performa drastis (Macro F1 $0,3725–0,3846$), membuktikan terjadinya *domain mismatch* antara bahasa *pretraining* Twitter dengan gaya penulisan ulasan pariwisata Google Maps.
* **Full Fine-Tuning (Run 5–6):** Melampaui *baseline* (Macro F1 hingga $0,7481$), tetapi menunjukkan indikasi awal *overfitting* (*validation loss* meningkat sejak *epoch* ke-2/ke-3) akibat pembaruan seluruh 110 juta parameter pada data latih yang terbatas.
* **Frozen-6 Layers + Dropout 0,1 (Run 9):** Membekukan `layer 0–5` menjaga pengetahuan sintaksis dan morfologi dasar bahasa Indonesia, sementara `layer 6–11` beradaptasi dengan konteks pariwisata. Pelonggaran *dropout* menjadi `0,1` meningkatkan kapasitas belajar *classifier head*, sedangkan *dropout* `0,5` (Run 11–12) menyebabkan *underfitting*.

### 2. Paradoks Efek Emoji (*The Emoji Effect Paradox*)
* Pada **TF-IDF + Logistic Regression**, emoji berfungsi sebagai fitur diskrit independen sehingga mode *Keep Emoji* meningkatkan Macro F1 sebesar $+0,0033$ poin.
* Pada **IndoBERTweet yang di-*fine-tune***, mode *No Emoji* justru unggul konsisten sebesar $0,0187$ hingga $0,0513$ poin Macro F1. Total 1.173 token emoji terdeteksi pada korpus; sementara emoji pada kelas Positif cenderung eksplisit, emoji pada kelas Negatif memiliki spektrum makna yang ambigu (kepasrahan, kekecewaan, hingga sarkasme) dan menambah kompleksitas urutan token tanpa memberikan sinyal sentimen baru di luar kata-kata evaluatif.

### 3. Error Analysis & Pengujian Demo
* **Kalibrasi Confidence Score:** Pada evaluasi 479 sampel *test set* menggunakan Run 9 (403 prediksi benar dan 76 prediksi salah), rata-rata *confidence score* (probabilitas Softmax tertinggi) pada prediksi benar adalah **79,8%**, jauh lebih tinggi dibandingkan prediksi salah sebesar **59,3%**.
* **Kategori Kesalahan Dominan:** Berdasarkan kategorisasi heuristik pada 76 kasus error, kesalahan terbanyak berasal dari **False Polar** (22 kasus / 28,9%) dan **Neutral Ambigu** (17 kasus / 22,4%), sedangkan kasus **Sarkasme / Ironi** hanya berjumlah 3 kasus (3,9%).
* **Validasi Demo:** Model mampu menangani *code-switching* (*"View nya epic banget, worth every rupiah!"* $\rightarrow$ **POSITIF**, *confidence* 95,7%) dan dominasi keluhan pada sentimen campuran (*"Jelek banget gilakk masak toiletnya kotor... meski petugasnya ramah lah ya"* $\rightarrow$ **NEGATIF**, *confidence* 75,0%), namun masih terkendala pada kalimat sarkasme implisit dan ulasan deskriptif ambigu.

---

## Keterbatasan Penelitian

1. **Metodologis:** Penggunaan *Majority Voting* untuk mengonversi 6 label aspek ABSA menjadi 1 label dokumen berpotensi menimbulkan *annotation noise* pada ulasan dengan sentimen campuran (*mixed sentiment*). Selain itu, kamus `NORM_DICT` bersifat statis sehingga memerlukan pembaruan berkala untuk menangkap kosakata *slang* baru.
2. **Linguistik:** Model masih menghadapi tantangan dalam memahami kosakata bahasa daerah yang jarang muncul di data latih (seperti kata bahasa Jawa *"medeni"*) serta ekspresi sarkasme/ironi.
3. **Data:** Cakupan data terbatas pada ulasan berbahasa Indonesia di Google Maps periode 2021–2025 dengan distribusi ulasan yang tidak merata antar destinasi wisata di Pulau Jawa.