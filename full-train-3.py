import os
import shutil
import random
import cv2
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import albumentations as A

class YoloDataPipelineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 OBB Dataset Pipeline Tool (Enhanced UI)")
        self.root.geometry("700x820")
        self.root.configure(bg="#F4F6F9")

        self.project_dir = tk.StringVar()
        
        self.setup_styles()
        self.setup_header()
        
        self.notebook = ttk.Notebook(root, style="Custom.TNotebook")
        self.notebook.pack(expand=True, fill="both", padx=15, pady=(5, 15))

        self.tab_grab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.tab_aug = ttk.Frame(self.notebook, style="Card.TFrame")
        self.tab_split = ttk.Frame(self.notebook, style="Card.TFrame")
        self.tab_train = ttk.Frame(self.notebook, style="Card.TFrame")

        self.notebook.add(self.tab_grab, text="  1. Grab Dataset  ")
        self.notebook.add(self.tab_aug, text="  2. Augmentasi OBB  ")
        self.notebook.add(self.tab_split, text="  3. Split & YAML  ")
        self.notebook.add(self.tab_train, text="  4. Training Model  ")

        self.setup_tab_grab()
        self.setup_tab_aug()
        self.setup_tab_split()
        self.setup_tab_train()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("vista" if "vista" in self.style.theme_names() else "default")
        
        self.style.configure(".", background="#F4F6F9", font=("Segoe UI", 10))
        self.style.configure("Card.TFrame", background="#FFFFFF")
        
        self.style.configure("Custom.TNotebook", background="#F4F6F9", borderwidth=0)
        self.style.configure("Custom.TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 6], background="#E2E8F0", foreground="#4A5568")
        self.style.map("Custom.TNotebook.Tab", background=[("selected", "#FFFFFF")], foreground=[("selected", "#2B6CB0")])

        self.style.configure("TLabelframe", background="#FFFFFF", bordercolor="#E2E8F0", borderwidth=1, relief="solid")
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground="#2D3748", background="#FFFFFF")

        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#3182CE", foreground="white", padding=8)
        self.style.map("Primary.TButton", background=[("active", "#2B6CB0")])
        
        self.style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), background="#38A169", foreground="white", padding=10)
        self.style.map("Success.TButton", background=[("active", "#2F855A")])
        
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background="#DD6B20", foreground="white", padding=10)
        self.style.map("Accent.TButton", background=[("active", "#C05621")])
        
        self.style.configure("Train.TButton", font=("Segoe UI", 11, "bold"), background="#E53E3E", foreground="white", padding=12)
        self.style.map("Train.TButton", background=[("active", "#C53030")])

        self.style.configure("TEntry", fieldbackground="#F7FAFC", bordercolor="#CBD5E0", padding=5)

    def setup_header(self):
        header_frame = tk.Frame(self.root, bg="#FFFFFF", bd=0, highlightbackground="#E2E8F0", highlightthickness=1)
        header_frame.pack(fill="x", padx=15, pady=15)
        
        inner_frame = tk.Frame(header_frame, bg="#FFFFFF", pady=15, padx=15)
        inner_frame.pack(fill="x")
        
        tk.Label(inner_frame, text="WORKSPACE DASHBOARD", font=("Segoe UI", 12, "bold"), fg="#1A365D", bg="#FFFFFF").pack(anchor="w", pady=(0,5))
        tk.Label(inner_frame, text="Tentukan folder project utama untuk mengotomatisasi seluruh alur kerja dataset.", font=("Segoe UI", 9), fg="#718096", bg="#FFFFFF").pack(anchor="w", pady=(0,10))
        
        dir_frame = tk.Frame(inner_frame, bg="#FFFFFF")
        dir_frame.pack(fill="x")
        
        self.entry_proj = ttk.Entry(dir_frame, textvariable=self.project_dir, width=55, font=("Segoe UI", 10))
        self.entry_proj.pack(side="left", padx=(0, 10), ipady=3, expand=True, fill="x")
        
        btn_proj = ttk.Button(dir_frame, text="Buat / Pilih Project", style="Primary.TButton", command=self.set_project_dir)
        btn_proj.pack(side="right")

    def set_project_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.project_dir.set(folder)
            folders = ['1_Raw_Images', '2_Exported_AnyLabeling', '3_Augmented', '4_YOLO_Ready', '5_Training_Results']
            for f in folders:
                os.makedirs(os.path.join(folder, f), exist_ok=True)
            messagebox.showinfo("Sukses", f"Workspace Project Berhasil Dibuat!\nStruktur folder 1 s/d 5 siap digunakan.\n\nLokasi:\n{folder}")

    def check_project_ready(self):
        if not self.project_dir.get():
            messagebox.showwarning("Workspace Kosong", "Silakan pilih atau buat Folder Project Utama terlebih dahulu di bagian atas!")
            return False
        return True

    # ================= TAB 1: GRAB DATASET =================
    def setup_tab_grab(self):
        container = tk.Frame(self.tab_grab, bg="#FFFFFF", padx=20, pady=20)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="1. Pengumpulan Gambar Mentah (Grab)", font=("Segoe UI", 13, "bold"), fg="#2B6CB0", bg="#FFFFFF").pack(anchor="w", pady=(0,5))
        tk.Label(container, text="Ambil dataset langsung melalui Kamera Lokal (Webcam) atau IP Cam (RTSP URL).", font=("Segoe UI", 9), fg="#718096", bg="#FFFFFF").pack(anchor="w", pady=(0,20))

        config_frame = ttk.LabelFrame(container, text=" Konfigurasi Kelas & Kamera ", padding=15)
        config_frame.pack(fill="x", pady=5)

        tk.Label(config_frame, text="Daftar Nama Kelas / Objek (pisahkan dengan koma):", font=("Segoe UI", 9, "bold"), bg="#FFFFFF", fg="#4A5568").pack(anchor="w", pady=(0,5))
        self.entry_cls = ttk.Entry(config_frame, font=("Segoe UI", 10))
        self.entry_cls.insert(0, "truk, mobil")
        self.entry_cls.pack(fill="x", pady=(0,15), ipady=3)

        tk.Label(config_frame, text="Sumber Aliran Video (0 = Webcam Internal, atau Masukkan URL RTSP):", font=("Segoe UI", 9, "bold"), bg="#FFFFFF", fg="#4A5568").pack(anchor="w", pady=(0,5))
        self.entry_cam = ttk.Entry(config_frame, font=("Segoe UI", 10))
        self.entry_cam.insert(0, "0")
        self.entry_cam.pack(fill="x", pady=(0,5), ipady=3)

        info_box = tk.Frame(container, bg="#EBF8FF", bd=0, highlightbackground="#BEE3F8", highlightthickness=1)
        info_box.pack(fill="x", pady=20)
        
        info_text = (
            "PETUNJUK TOMBOL JENDELA KAMERA:\n"
            "• Tekan [ S ] pada keyboard untuk memotret/simpan gambar mentah ke folder '1_Raw_Images'\n"
            "• Tekan [ Q ] pada keyboard untuk menghentikan dan menutup aliran kamera."
        )
        tk.Label(info_box, text=info_text, font=("Segoe UI", 9, "italic"), fg="#2B6CB0", bg="#EBF8FF", justify="left", padx=15, pady=10).pack(anchor="w")

        btn_start_grab = ttk.Button(container, text="SIMPAN KONFIGURASI & BUKA KAMERA", style="Success.TButton", command=self.run_grab)
        btn_start_grab.pack(fill="x", side="bottom", pady=10)

    def run_grab(self):
        if not self.check_project_ready(): return
        
        classes_text = self.entry_cls.get()
        classes_list = [c.strip() for c in classes_text.split(",") if c.strip()]
        if not classes_list:
            messagebox.showwarning("Parameter Kurang", "Harap isi minimal satu nama kelas objek!")
            return

        with open(os.path.join(self.project_dir.get(), "classes.txt"), "w") as f:
            f.write("\n".join(classes_list))
            
        source = self.entry_cam.get()
        source = int(source) if source.isdigit() else source
        save_dir = os.path.join(self.project_dir.get(), '1_Raw_Images')
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Koneksi Gagal", "Sumber video/kamera tidak dapat diakses! Pastikan ID atau URL benar.")
            return

        count = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
        messagebox.showinfo("Kamera Siap", "Jendela kamera akan terbuka.\n\nFokus ke layar kamera:\nTekan 'S' untuk Capture.\nTekan 'Q' untuk Selesai.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow("Capture System - Press [S] to Save / [Q] to Quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                count += 1
                filename = os.path.join(save_dir, f"img_{count:04d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[Captured] Disimpan: {filename}")
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # ================= TAB 2: AUGMENTASI =================
    def setup_tab_aug(self):
        container = tk.Frame(self.tab_aug, bg="#FFFFFF", padx=20, pady=20)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="2. Efek Augmentasi Real-Time OBB", font=("Segoe UI", 13, "bold"), fg="#2B6CB0", bg="#FFFFFF").pack(anchor="w", pady=(0,5))
        tk.Label(container, text="Melipatgandakan variasi data dengan transformasi koordinat matematika miring secara presisi.", font=("Segoe UI", 9), fg="#718096", bg="#FFFFFF").pack(anchor="w", pady=(0,15))

        warn_box = tk.Frame(container, bg="#FFF5F5", bd=0, highlightbackground="#FED7D7", highlightthickness=1)
        warn_box.pack(fill="x", pady=(0,15))
        
        warn_text = (
            "PENTING: Lakukan anotasi di X-AnyLabeling terlebih dahulu, lalu simpan hasil ekspor\n"
            "berformat 'YOLO OBB' ke dalam folder '2_Exported_AnyLabeling' sebelum menekan tombol."
        )
        tk.Label(warn_box, text=warn_text, font=("Segoe UI", 9), fg="#C53030", bg="#FFF5F5", justify="left", padx=12, pady=8).pack(anchor="w")

        fx_frame = ttk.LabelFrame(container, text=" Pilih Transformasi Efek ", padding=15)
        fx_frame.pack(fill="x", pady=5)

        self.var_flip = tk.BooleanVar(value=True)
        self.var_rot = tk.BooleanVar(value=True)
        self.var_bright = tk.BooleanVar(value=True)
        self.var_noise = tk.BooleanVar(value=False)

        cb_grid = tk.Frame(fx_frame, bg="#FFFFFF")
        cb_grid.pack(fill="x", pady=5)
        ttk.Checkbutton(cb_grid, text="Horizontal Flip (Cermin Pasif)", variable=self.var_flip).grid(row=0, column=0, sticky="w", padx=20, pady=8)
        ttk.Checkbutton(cb_grid, text="Rotasi Sudut Bebas OBB", variable=self.var_rot).grid(row=0, column=1, sticky="w", padx=20, pady=8)
        ttk.Checkbutton(cb_grid, text="Kecerahan & Kontras Dinamis", variable=self.var_bright).grid(row=1, column=0, sticky="w", padx=20, pady=8)
        ttk.Checkbutton(cb_grid, text="Gaussian Noise (Bintik Sensor)", variable=self.var_noise).grid(row=1, column=1, sticky="w", padx=20, pady=8)

        param_frame = ttk.LabelFrame(container, text=" Pengaturan Nilai Parameter Efek ", padding=15)
        param_frame.pack(fill="x", pady=10)
        param_frame.columnconfigure(1, weight=1)

        tk.Label(param_frame, text="Batas Rotasi Maksimal (± Derajat):", bg="#FFFFFF").grid(row=0, column=0, sticky="w", pady=5)
        self.entry_rot_limit = ttk.Entry(param_frame, width=12)
        self.entry_rot_limit.insert(0, "30")
        self.entry_rot_limit.grid(row=0, column=1, sticky="e", pady=5)

        tk.Label(param_frame, text="Peluang Munculnya Efek (Skala 0.1 s/d 1.0):", bg="#FFFFFF").grid(row=1, column=0, sticky="w", pady=5)
        self.entry_aug_prob = ttk.Entry(param_frame, width=12)
        self.entry_aug_prob.insert(0, "0.5")
        self.entry_aug_prob.grid(row=1, column=1, sticky="e", pady=5)

        tk.Label(param_frame, text="Jumlah Output Variasi (Per 1 Gambar Asli):", bg="#FFFFFF").grid(row=2, column=0, sticky="w", pady=5)
        self.entry_aug_count = ttk.Entry(param_frame, width=12)
        self.entry_aug_count.insert(0, "3")
        self.entry_aug_count.grid(row=2, column=1, sticky="e", pady=5)

        btn_start_aug = ttk.Button(container, text="PROSES MULTIPLIKASI DATA (AUGMENTASI)", style="Accent.TButton", command=self.run_aug)
        btn_start_aug.pack(fill="x", side="bottom", pady=10)

    def run_aug(self):
        if not self.check_project_ready(): return
        
        proj_dir = self.project_dir.get()
        raw_dir = os.path.join(proj_dir, '1_Raw_Images')
        export_dir = os.path.join(proj_dir, '2_Exported_AnyLabeling')
        output_dir = os.path.join(proj_dir, '3_Augmented')
        
        out_img_dir = os.path.join(output_dir, 'images')
        out_lbl_dir = os.path.join(output_dir, 'labels')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        txt_files = [f for f in os.listdir(export_dir) if f.endswith('.txt')]
        if not txt_files:
            messagebox.showerror("Data Label Hilang", "Tidak ada berkas anotasi (.txt) di dalam folder '2_Exported_AnyLabeling'.")
            return

        try: rot_limit = int(self.entry_rot_limit.get())
        except ValueError: rot_limit = 30
        
        try: aug_prob = float(self.entry_aug_prob.get())
        except ValueError: aug_prob = 0.5
        
        try: num_aug = int(self.entry_aug_count.get())
        except ValueError: num_aug = 1

        transforms = []
        if self.var_flip.get(): transforms.append(A.HorizontalFlip(p=aug_prob))
        if self.var_rot.get(): transforms.append(A.Rotate(limit=rot_limit, p=aug_prob, border_mode=cv2.BORDER_CONSTANT))
        if self.var_bright.get(): transforms.append(A.RandomBrightnessContrast(p=aug_prob))
        if self.var_noise.get(): transforms.append(A.GaussNoise(p=aug_prob))

        if not transforms:
            messagebox.showwarning("Opsi Kosong", "Pilih minimal satu efek transformasi augmentasi!")
            return

        transform = A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        berhasil = 0
        for txt_name in txt_files:
            base_name = os.path.splitext(txt_name)[0]
            img_path = os.path.join(export_dir, base_name + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(raw_dir, base_name + '.jpg')
                
            if not os.path.exists(img_path): continue
                
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]

            classes = []
            keypoints = []
            with open(os.path.join(export_dir, txt_name), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        classes.append(parts[0])
                        pts = [float(p) for p in parts[1:]]
                        keypoints.extend([
                            (pts[0]*w, pts[1]*h), (pts[2]*w, pts[3]*h),
                            (pts[4]*w, pts[5]*h), (pts[6]*w, pts[7]*h)
                        ])

            cv2.imwrite(os.path.join(out_img_dir, f"{base_name}.jpg"), img)
            shutil.copy(os.path.join(export_dir, txt_name), os.path.join(out_lbl_dir, txt_name))

            if len(keypoints) > 0:
                for i in range(num_aug):
                    augmented = transform(image=img, keypoints=keypoints)
                    aug_img = augmented['image']
                    aug_kps = augmented['keypoints']

                    new_base = f"{base_name}_aug{i}"
                    cv2.imwrite(os.path.join(out_img_dir, f"{new_base}.jpg"), aug_img)

                    with open(os.path.join(out_lbl_dir, f"{new_base}.txt"), 'w') as f:
                        idx = 0
                        for cls in classes:
                            kp = aug_kps[idx:idx+4]
                            idx += 4
                            norm_kp = []
                            for px, py in kp:
                                norm_kp.extend([max(0, min(1, px/w)), max(0, min(1, py/h))])
                            
                            line_str = f"{cls} " + " ".join([f"{v:.6f}" for v in norm_kp]) + "\n"
                            f.write(line_str)
            berhasil += 1
            
        messagebox.showinfo("Augmentasi Sukses", f"Proses Selesai!\n\n{berhasil} data asli berhasil diekspansi menjadi total {berhasil * (num_aug + 1)} file citra teraugmentasi.")

    # ================= TAB 3: SPLIT & YAML =================
    def setup_tab_split(self):
        container = tk.Frame(self.tab_split, bg="#FFFFFF", padx=20, pady=20)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="3. Distribusi Data (Split) & File YAML", font=("Segoe UI", 13, "bold"), fg="#2B6CB0", bg="#FFFFFF").pack(anchor="w", pady=(0,5))
        tk.Label(container, text="Membagi database citra secara acak menjadi porsi Train, Validation, dan Test secara teratur.", font=("Segoe UI", 9), fg="#718096", bg="#FFFFFF").pack(anchor="w", pady=(0,20))

        ratio_frame = ttk.LabelFrame(container, text=" Atur Proporsi Pembagian Rasio Data (%) ", padding=20)
        ratio_frame.pack(fill="x", pady=10)
        ratio_frame.columnconfigure(1, weight=1)

        tk.Label(ratio_frame, text="Porsi Belajar Mandiri (Train %):", bg="#FFFFFF").grid(row=0, column=0, sticky="w", pady=8)
        self.e_tr = ttk.Entry(ratio_frame, width=15, font=("Segoe UI", 10))
        self.e_tr.insert(0, "80")
        self.e_tr.grid(row=0, column=1, sticky="e", pady=8)

        tk.Label(ratio_frame, text="Porsi Validasi Evaluasi (Val %):", bg="#FFFFFF").grid(row=1, column=0, sticky="w", pady=8)
        self.e_vl = ttk.Entry(ratio_frame, width=15, font=("Segoe UI", 10))
        self.e_vl.insert(0, "10")
        self.e_vl.grid(row=1, column=1, sticky="e", pady=8)

        tk.Label(ratio_frame, text="Porsi Uji Akhir Buta (Test %):", bg="#FFFFFF").grid(row=2, column=0, sticky="w", pady=8)
        self.e_ts = ttk.Entry(ratio_frame, width=15, font=("Segoe UI", 10))
        self.e_ts.insert(0, "10")
        self.e_ts.grid(row=2, column=1, sticky="e", pady=8)

        info_yaml = tk.Frame(container, bg="#F7FAFC", bd=0, highlightbackground="#E2E8F0", highlightthickness=1)
        info_yaml.pack(fill="x", pady=20)
        
        yaml_text = (
            "Catatan Sistem:\n"
            "Aplikasi akan membaca daftar objek dari berkas 'classes.txt' di folder proyek secara otomatis\n"
            "untuk mengompilasi susunan file konfigurasi dataset.yaml."
        )
        tk.Label(info_yaml, text=yaml_text, font=("Segoe UI", 9), fg="#4A5568", bg="#F7FAFC", justify="left", padx=15, pady=12).pack(anchor="w")

        btn_start_split = ttk.Button(container, text="EKSEKUSI PEMBAGIAN DATA & GENERATE YAML", style="Primary.TButton", command=self.run_split)
        btn_start_split.pack(fill="x", side="bottom", pady=10)

    def run_split(self):
        if not self.check_project_ready(): return
        
        src_dir = os.path.join(self.project_dir.get(), '3_Augmented')
        out_dir = os.path.join(self.project_dir.get(), '4_YOLO_Ready')
        classes_file = os.path.join(self.project_dir.get(), "classes.txt")
        
        if not os.path.exists(os.path.join(src_dir, 'images')) or not os.listdir(os.path.join(src_dir, 'images')):
            messagebox.showerror("Dataset Kosong", "Folder data '3_Augmented/images' kosong! Jalankan kompresi efek augmentasi terlebih dahulu.")
            return

        try:
            tr_v = float(self.e_tr.get())
            vl_v = float(self.e_vl.get())
            ts_v = float(self.e_ts.get())
        except ValueError:
            messagebox.showerror("Error Angka", "Nilai input rasio pembagian harus berupa angka murni!")
            return

        if tr_v + vl_v + ts_v != 100:
            messagebox.showerror("Kalkulasi Salah", f"Total penjumlahan persentase rasio saat ini ({tr_v+vl_v+ts_v}%) HARUS tepat 100%!")
            return

        if not os.path.exists(classes_file):
            messagebox.showerror("File Absen", "Berkas 'classes.txt' tidak ditemukan di ruang proyek utama!")
            return

        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]

        splits = ['train', 'val', 'test']
        for s in splits:
            os.makedirs(os.path.join(out_dir, 'images', s), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'labels', s), exist_ok=True)

        all_files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(src_dir, 'images')) if f.endswith('.jpg')]
        random.shuffle(all_files)
        total = len(all_files)
        
        tr_idx = int(total * (tr_v / 100))
        vl_idx = tr_idx + int(total * (vl_v / 100))
        
        dist = {'train': all_files[:tr_idx], 'val': all_files[tr_idx:vl_idx], 'test': all_files[vl_idx:]}

        for split_name, files in dist.items():
            for f in files:
                shutil.copy(os.path.join(src_dir, 'images', f + '.jpg'), os.path.join(out_dir, 'images', split_name, f + '.jpg'))
                lbl_path = os.path.join(src_dir, 'labels', f + '.txt')
                if os.path.exists(lbl_path):
                    shutil.copy(lbl_path, os.path.join(out_dir, 'labels', split_name, f + '.txt'))

        yaml_path = os.path.join(out_dir, "dataset.yaml")
        yaml_content = f"path: {out_dir.replace(chr(92), '/')}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
        for i, c in enumerate(classes): yaml_content += f"  {i}: {c}\n"
        
        with open(yaml_path, 'w') as f: f.write(yaml_content)
        messagebox.showinfo("Split Sukses", f"Pembagian Dataset Tuntas!\n\nStruktur folder '4_YOLO_Ready' beserta skrip 'dataset.yaml' siap dipasok ke mesin training.")

    # ================= TAB 4: TRAINING =================
    def setup_tab_train(self):
        container = tk.Frame(self.tab_train, bg="#FFFFFF", padx=20, pady=20)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="4. Panel Kendali Utama Pelatihan Model", font=("Segoe UI", 13, "bold"), fg="#2B6CB0", bg="#FFFFFF").pack(anchor="w", pady=(0,5))
        tk.Label(container, text="Konfigurasi parameter Hyperparameters jaringan neural network YOLOv8-OBB secara dinamis.", font=("Segoe UI", 9), fg="#718096", bg="#FFFFFF").pack(anchor="w", pady=(0,15))

        hp_frame = ttk.LabelFrame(container, text=" Konfigurasi Jaringan & Hyperparameters ", padding=15)
        hp_frame.pack(fill="x", pady=5)
        
        hp_frame.columnconfigure(1, weight=1)
        hp_frame.columnconfigure(3, weight=1)

        tk.Label(hp_frame, text="Arsitektur Basis:", bg="#FFFFFF").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.entry_model = ttk.Entry(hp_frame, width=12)
        self.entry_model.insert(0, "yolov8n-obb.pt")
        self.entry_model.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        tk.Label(hp_frame, text="Maksimal Epoch:", bg="#FFFFFF").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.entry_epochs = ttk.Entry(hp_frame, width=12)
        self.entry_epochs.insert(0, "50")
        self.entry_epochs.grid(row=0, column=3, sticky="w", padx=10, pady=5)

        tk.Label(hp_frame, text="Resolusi Citra:", bg="#FFFFFF").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_imgsz = ttk.Entry(hp_frame, width=12)
        self.entry_imgsz.insert(0, "416")
        self.entry_imgsz.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        tk.Label(hp_frame, text="Batch Size:", bg="#FFFFFF").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.entry_batch = ttk.Entry(hp_frame, width=12)
        self.entry_batch.insert(0, "2")
        self.entry_batch.grid(row=1, column=3, sticky="w", padx=10, pady=5)

        tk.Label(hp_frame, text="Batas Patience:", bg="#FFFFFF").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.entry_patience = ttk.Entry(hp_frame, width=12)
        self.entry_patience.insert(0, "15")
        self.entry_patience.grid(row=2, column=1, sticky="w", padx=10, pady=5)

        tk.Label(hp_frame, text="CPU Workers:", bg="#FFFFFF").grid(row=2, column=2, sticky="w", padx=5, pady=5)
        self.entry_workers = ttk.Entry(hp_frame, width=12)
        self.entry_workers.insert(0, "0")
        self.entry_workers.grid(row=2, column=3, sticky="w", padx=10, pady=5)

        tk.Label(hp_frame, text="Device Target:", bg="#FFFFFF").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.entry_device = ttk.Entry(hp_frame, width=12)
        self.entry_device.insert(0, "cpu")
        self.entry_device.grid(row=3, column=1, sticky="w", padx=10, pady=5)

        self.prog_box = ttk.LabelFrame(container, text=" Status & Indikator Progress Pelatihan ", padding=15)
        self.prog_box.pack(fill="x", pady=15)

        self.lbl_train_status = tk.Label(self.prog_box, text="Status Workspace: Menunggu Perintah Pelatihan...", font=("Segoe UI", 10, "bold"), fg="#4A5568", bg="#FFFFFF")
        self.lbl_train_status.pack(anchor="w", pady=(0, 8))

        self.pbar = ttk.Progressbar(self.prog_box, orient="horizontal", mode="determinate")
        self.pbar.pack(fill="x", pady=5)
        self.pbar['value'] = 0

        self.btn_train = ttk.Button(container, text="LAUNCH NEURAL NETWORK TRAINING", style="Train.TButton", command=self.start_train_thread)
        self.btn_train.pack(fill="x", side="bottom", pady=5)

    def start_train_thread(self):
        if not self.check_project_ready(): return
        
        self.btn_train.config(state=tk.DISABLED)
        self.lbl_train_status.config(text="Status Workspace: Menginisialisasi modul arsitektur YOLO...", fg="#DD6B20")
        self.pbar['value'] = 0
        
        threading.Thread(target=self._run_train, daemon=True).start()

    def update_progress_ui(self, ep, tot, pct):
        self.lbl_train_status.config(text=f"Status Workspace: Memproses Formasi Epoch [ {ep} / {tot} ] ({pct}%)", fg="#3182CE")
        self.pbar['value'] = pct

    def _run_train(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.root.after(0, lambda: messagebox.showerror("Library Missing", "Modul 'ultralytics' tidak ditemukan di venv ini!"))
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.lbl_train_status.config(text="Status Workspace: Kegagalan sistem memicu interupsi.", fg="#E53E3E"))
            return

        yaml_path = os.path.join(self.project_dir.get(), '4_YOLO_Ready', 'dataset.yaml')
        if not os.path.exists(yaml_path):
            self.root.after(0, lambda: messagebox.showerror("YAML Hilang", "Berkas konfigurasi 'dataset.yaml' absen di folder 4_YOLO_Ready!"))
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL))
            return

        model_name = self.entry_model.get()
        epochs = int(self.entry_epochs.get())
        imgsz = int(self.entry_imgsz.get())
        batch = int(self.entry_batch.get())
        patience = int(self.entry_patience.get())
        workers = int(self.entry_workers.get())
        device = self.entry_device.get()
        
        target_dir = os.path.join(self.project_dir.get(), '5_Training_Results')
        os.makedirs(target_dir, exist_ok=True)

        try:
            model = YOLO(model_name)
            
            def on_epoch_end(trainer):
                ep = trainer.epoch + 1
                tot = trainer.epochs
                pct = int((ep / tot) * 100)
                self.root.after(0, lambda: self.update_progress_ui(ep, tot, pct))

            model.add_callback("on_fit_epoch_end", on_epoch_end)
            
            model.train(
                data=yaml_path, 
                epochs=epochs, 
                imgsz=imgsz, 
                batch=batch, 
                device=device, 
                patience=patience, 
                workers=workers,
                project=target_dir, 
                name="Sesi_Training"
            )
            
            self.root.after(0, lambda: self.lbl_train_status.config(text="Status Workspace: Sukses Total! Bobot Model Tersimpan.", fg="#38A169"))
            self.root.after(0, lambda: self.pbar.config(value=100))
            self.root.after(0, lambda: messagebox.showinfo("Pelatihan Berhasil", "Siklus Pembelajaran Selesai!\nBobot akurasi terbaik disimpan rapi di folder '5_Training_Results/Sesi_Training'."))
            
        except Exception as e:
            self.root.after(0, lambda: self.lbl_train_status.config(text="Status Workspace: Interupsi Error Eksternal Terdeteksi.", fg="#E53E3E"))
            self.root.after(0, lambda: messagebox.showerror("Gagal Operasi", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloDataPipelineApp(root)
    root.mainloop()
