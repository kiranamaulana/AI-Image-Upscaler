# AI-Image-Upscaler
AI Image Upscaler
# 📸 NunoAI Upscaler - 4K Ultra HD

**NunoAI Upscaler** adalah aplikasi desktop berbasis Artificial Intelligence (AI) yang dirancang untuk meningkatkan resolusi foto (Upscaling) dan menjernihkan foto buram secara otomatis hingga kualitas 4K.

## 🚀 Fitur Utama
- **AI Enhancement:** Menggunakan arsitektur RRDBNet & Real-ESRGAN yang canggih.
- **Ultra HD/4K:** Tingkatkan resolusi gambar hingga 4 kali lipat tanpa pecah.
- **Simple UI:** Antarmuka minimalis dan mudah digunakan.
- **Privacy First:** Proses dilakukan secara lokal, foto Anda tidak akan diupload ke internet.

## 🛠️ Requirements & Setup
1. Pastikan Anda memiliki Python 3.8+.
2. Instal library yang dibutuhkan:
   ```bash
   pip install realesrgan basicsr opencv-python-headless

   Penting: Pastikan file model RealESRGAN_x4plus.pth berada di folder yang sama dengan script.

📥 Download Model
Jika file model belum ada, 
silakan unduh di sini:<a href="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"> RealESRGAN_x4plus.pth</a>
