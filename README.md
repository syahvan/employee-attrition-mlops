# Submission 2: Employee Attrition Prediction

Nama: Syahvan Alviansyah Diva Ritonga

Username dicoding: syahvan

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/) |
| Masalah | Employee attrition adalah kondisi ketika karyawan meninggalkan perusahaan, baik secara sukarela maupun tidak. Kondisi ini dapat menyebabkan kerugian finansial dan produktivitas yang signifikan bagi perusahaan, terutama jika karyawan yang keluar adalah karyawan yang berpengalaman atau memiliki keterampilan yang sulit digantikan. Mengurangi tingkat attrition menjadi prioritas penting bagi banyak perusahaan untuk menjaga stabilitas dan efisiensi operasional. |
| Solusi machine learning | Dari permasalahan diatas dengan memanfaatkan teknologi, machine learning menjadi salah satu solusi untuk membantu mengurangi tingkat attrition yang tinggi. Dengan sebuah sistem prediksi employee attrition, diharapkan perusahaan dapat mengidentifikasi karyawan yang berpotensi untuk keluar dan mengambil tindakan preventif untuk mempertahankan mereka. |
| Metode pengolahan | Data yang digunakan pada proyek ini terdapat dua tipe data, yaitu data kategorikal dan numerik. Metode yang digunakan untuk mengelola data tersebut yaitu mentransformasikan data kategorikal menjadi bentuk one-hot encoding dan menormalisasikan data numerik kedalam range data yang sama. |
| Arsitektur model | Model yang dibangun cukup sederhana hanya menggunakan Dense layer dan Dropout layer sebagai hidden layer pada model neural network dan memiliki 1 output layer. |
| Metrik evaluasi | Metrik yang digunakan pada model yaitu AUC, Precision, Recall, BinaryAccuracy, TruePositive, FalsePositive, TrueNegative, FalseNegative untuk mengevaluasi performa model sebuah klasifikasi. |
| Performa model | Model yang dibuat menghasilkan performa yang cukup baik dalam memberikan sebuah prediksi dan dari pelatihan yang dilakukan menghasilkan binary_accuracy sebesar 86% dan val_binary_accuracy sebesar 82%. Hasil seperti ini sudah cukup baik untuk sebuah sistem klasifikasi namun masih bisa ditingkatkan lagi. |
| Opsi deployment | Proyek machine learning ini dideploy menggunakan salah satu platform as a service yaitu Railway yang menyediakan layanan gratis untuk mendeploy sebuah proyek. |
| Web app | <https://attrition-production.up.railway.app/v1/models/employee-attrition-model> |
| Monitoring | Monitoring pada proyek ini dapat dilakukan dengan menggunakan layanan open-source yaitu Prometheus. Contohnya setiap perubahan jumlah request yang dilakukan kepada sistem ini dapat dimonitoring dengan baik dan dapat menampilkan status dari setiap request yang diterima. |