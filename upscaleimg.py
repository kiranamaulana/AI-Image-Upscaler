import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2

class NunoUpscaler:
    def __init__(self, root):
        self.root = root
        self.root.title("NunoAI Upscaler - 4K Ultra HD")
        self.root.geometry("700x500")
        self.root.configure(bg="#0f172a")

        self.setup_ui()

    def setup_ui(self):
        # Header Premium
        header = tk.Frame(self.root, bg="#1e293b")
        header.pack(fill="x")
        tk.Label(header, text="AI IMAGE UPSCALER 4K", font=("Segoe UI", 18, "bold"), fg="#38bdf8", bg="#1e293b").pack(pady=15)

        self.main_f = tk.Frame(self.root, bg="#0f172a", padx=40, pady=20)
        self.main_f.pack(fill="both", expand=True)

        tk.Label(self.main_f, text="Select Blurry Image:", fg="#94a3b8", bg="#0f172a").pack(anchor="w")
        self.path_in = tk.StringVar()
        tk.Entry(self.main_f, textvariable=self.path_in, width=50).pack(pady=5)
        tk.Button(self.main_f, text="BROWSE IMAGE", command=self.select_file).pack(pady=5)

        tk.Label(self.main_f, text="Upscale Quality:", fg="#38bdf8", bg="#0f172a").pack(pady=(20,0))
        self.scale = ttk.Combobox(self.main_f, values=["2x (HD)", "4x (Ultra HD/4K)"], state="readonly")
        self.scale.set("4x (Ultra HD/4K)")
        self.scale.pack(pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(self.main_f, mode='indeterminate')
        self.progress.pack(fill="x", pady=30)

        self.btn_run = tk.Button(self.main_f, text="START AI UPSCALING", command=self.start_thread, 
                                 bg="#38bdf8", fg="#0f172a", font=("bold"), height=2)
        self.btn_run.pack(fill="x")

    def select_file(self):
        f = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.webp")])
        if f: self.path_in.set(f)

    def start_thread(self):
        if not self.path_in.get(): return
        self.btn_run.config(state="disabled", text="AI IS WORKING...")
        self.progress.start()
        threading.Thread(target=self.run_upscale, daemon=True).start()

    def run_upscale(self):
        try:
            input_path = self.path_in.get()
            # Cek apakah file model sudah ada di folder
            model_file = "RealESRGAN_x4plus.pth"
            
            if not os.path.exists(model_file):
                messagebox.showerror("Error", "File RealESRGAN_x4plus.pth tidak ditemukan di folder! Harap download dulu Mas.")
                return

            output_path = input_path.replace(".", "_4K_UltraHD.")
            
            # Pengaturan Model Tanpa Keyword yang Bikin Error
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_block=23, num_feat=64)
            
            # Kita panggil langsung menggunakan model_path ke file lokal
            upscaler = RealESRGANer(
                scale=4, 
                model_path=model_file, 
                model=model
            )
            
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("Gambar tidak terbaca! Cek format filenya Mas.")
                
            # Proses Menjernihkan
            output, _ = upscaler.enhance(img, outscale=4)
            
            # Simpan
            cv2.imwrite(output_path, output)
            messagebox.showinfo("Success", f"Fotonya sudah jernih Mas!\nDisimpan di: {output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"AI Error: {e}")
        finally:
            self.progress.stop()
            self.btn_run.config(state="normal", text="START AI UPSCALING")

if __name__ == "__main__":
    root = tk.Tk()
    app = NunoUpscaler(root)
    root.mainloop()