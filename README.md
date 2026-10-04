# BISINDO CNN-LSTM — Pipeline Pelatihan Model

Notebook Google Colab untuk pipeline lengkap pelatihan model pengenalan isyarat BISINDO menggunakan arsitektur hybrid 1D CNN-LSTM, dari data mentah hasil akuisisi sarung tangan IoT hingga model siap pakai.
---

![alt text](https://github.com/iosramgio/sketch_esp32/blob/main/public/Screenshot%202026-08-10%20185406.png)

---

## Alur Pipeline

1. **Load & Validasi Dataset** — memuat data sensor (flex + IMU) dan mengecek distribusi kelas
2. **Cleaning Data** — interpolasi linear untuk menangani glitch/nilai nol pada sensor
3. **Reshape Dataset** — mengubah data menjadi bentuk `(N, 100, 11)` (N sampel, 100 timestep, 11 fitur)
4. **Label Encoding** — encoding 10 kelas kosakata BISINDO
5. **Stratified Split** — pembagian data latih/validasi/uji dengan rasio 70/15/15
6. **Z-Score Normalization** — normalisasi fitur (scaler di-fit hanya dari data latih)
7. **Membangun Arsitektur Model** — CNN-LSTM (Conv1D → MaxPooling1D → LSTM → Dropout → Dense Softmax)
8. **Training** — pelatihan model dengan Early Stopping & ReduceLROnPlateau
9. **Evaluasi** — classification report, confusion matrix, dan pengujian latensi inferensi
10. **Simpan Model** — ekspor model final (`.keras`), scaler, dan label encoder

## Output

- `model_bisindo_final.keras` — model terlatih
- `scaler.pkl` — parameter normalisasi Z-Score
- `label_encoder.pkl` — encoder label kelas

## Requirements
- tensorflow
- numpy
- pandas
- scikit-learn
- matplotlib
- joblib
