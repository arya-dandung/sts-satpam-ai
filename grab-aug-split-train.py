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
        self.root.title("YOLOv8 OBB Dataset Pipeline Tool")
        self.root.geometry("600x750") # Tinggi disesuaikan untuk ekstra parameter
        self.root.resizable(False, False)

        self.project_dir = tk.StringVar()

        self.setup_header()
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.tab_grab = ttk.Frame(self.notebook)
        self.tab_aug = ttk.Frame(self.notebook)
        self.tab_split = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_grab, text=" 1. Grab Dataset ")
        self.notebook.add(self.tab_aug, text=" 2. Augmentasi OBB ")
        self.notebook.add(self.tab_split, text=" 3. Split & YAML ")
        self.notebook.add(self.tab_train, text=" 4. Training ")

        self.setup_tab_grab()
        self.setup_tab_aug()
        self.setup_tab_split()
        self.setup_tab_train()

    def setup_header(self):
        frame = tk.Frame(self.root, pady=10)
        frame.pack(fill="x")
        tk.Label(frame, text="Folder Project Utama:", font=("Arial", 10, "bold")).pack(anchor="w", padx=20)
        
        dir_frame = tk.Frame(frame)
        dir_frame.pack(fill="x", padx=20, pady=5)
        self.entry_proj = tk.Entry(dir_frame, textvariable=self.project_dir, width=60, state='readonly')
        self.entry_proj.pack(side="left", padx=(0, 10))
        tk.Button(dir_frame, text="Buat / Pilih Project", command=self.set_project_dir).pack(side="left")

    def set_project_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.project_dir.set(folder)
            folders = ['1_Raw_Images', '2_Exported_AnyLabeling', '3_Augmented', '4_YOLO_Ready']
            for f in folders:
                os.makedirs(os.path.join(folder, f), exist_ok=True)
            messagebox.showinfo("Project Disetel", f"Struktur folder berhasil dibuat di:\n{folder}")

    def check_project_ready(self):
        if not self.project_dir.get():
            messagebox.showwarning("Peringatan", "Pilih Folder Project Utama terlebih dahulu!")
            return False
        return True

    # ================= TAB 1: GRAB DATASET =================
    def setup_tab_grab(self):
        tk.Label(self.tab_grab, text="Setup Kelas & Grab Gambar", font=("Arial", 12, "bold")).pack(pady=15)
        
        tk.Label(self.tab_grab, text="Daftar Nama Objek/Kelas (pisahkan koma):").pack(anchor="w", padx=50)
        self.entry_cls = tk.Entry(self.tab_grab, width=50)
        self.entry_cls.insert(0, "truk, mobil")
        self.entry_cls.pack(pady=5, padx=50)

        frame_input = tk.Frame(self.tab_grab)
        frame_input.pack(pady=15)
        tk.Label(frame_input, text="Sumber Kamera (0 = WebCam / Link RTSP):").pack(anchor="w")
        self.entry_cam = tk.Entry(frame_input, width=50)
        self.entry_cam.insert(0, "0")
        self.entry_cam.pack(pady=5)
        
        tk.Button(self.tab_grab, text="SIMPAN KELAS & BUKA KAMERA", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), command=self.run_grab).pack(pady=15, fill="x", padx=50)

    def run_grab(self):
        if not self.check_project_ready(): return
        
        classes_text = self.entry_cls.get()
        classes_list = [c.strip() for c in classes_text.split(",") if c.strip()]
        with open(os.path.join(self.project_dir.get(), "classes.txt"), "w") as f:
            f.write("\n".join(classes_list))
            
        source = self.entry_cam.get()
        source = int(source) if source.isdigit() else source
        save_dir = os.path.join(self.project_dir.get(), '1_Raw_Images')
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Kamera/RTSP tidak dapat diakses!")
            return

        count = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
        messagebox.showinfo("Info", "Kamera terbuka.\nTekan S untuk memotret, Q untuk keluar.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow("Grab Dataset (S: Save, Q: Quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                count += 1
                filename = os.path.join(save_dir, f"img_{count:04d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Disimpan: {filename}")
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # ================= TAB 2: AUGMENTASI =================
    def setup_tab_aug(self):
        tk.Label(self.tab_aug, text="Parameter Augmentasi OBB", font=("Arial", 12, "bold")).pack(pady=10)
        
        frame_opts = tk.Frame(self.tab_aug)
        frame_opts.pack(pady=10)
        
        # Opsi Efek
        tk.Label(frame_opts, text="Pilih Efek:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(0,5))
        self.var_flip = tk.BooleanVar(value=True)
        self.var_rot = tk.BooleanVar(value=True)
        self.var_bright = tk.BooleanVar(value=True)
        self.var_noise = tk.BooleanVar(value=False)

        tk.Checkbutton(frame_opts, text="Horizontal Flip", variable=self.var_flip).grid(row=1, column=0, sticky="w", padx=10)
        tk.Checkbutton(frame_opts, text="Rotasi Acak", variable=self.var_rot).grid(row=1, column=1, sticky="w", padx=10)
        tk.Checkbutton(frame_opts, text="Kecerahan/Kontras", variable=self.var_bright).grid(row=2, column=0, sticky="w", padx=10)
        tk.Checkbutton(frame_opts, text="Bintik/Noise", variable=self.var_noise).grid(row=2, column=1, sticky="w", padx=10)
        
        # Parameter Khusus
        tk.Label(frame_opts, text="Batas Rotasi (± Derajat):").grid(row=3, column=0, sticky="w", padx=10, pady=(15,0))
        self.entry_rot_limit = tk.Entry(frame_opts, width=10)
        self.entry_rot_limit.insert(0, "30")
        self.entry_rot_limit.grid(row=3, column=1, sticky="w", padx=10, pady=(15,0))

        tk.Label(frame_opts, text="Peluang Efek Muncul (0.1 - 1.0):").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.entry_aug_prob = tk.Entry(frame_opts, width=10)
        self.entry_aug_prob.insert(0, "0.5")
        self.entry_aug_prob.grid(row=4, column=1, sticky="w", padx=10, pady=5)

        # Count
        frame_count = tk.Frame(self.tab_aug)
        frame_count.pack(pady=10)
        tk.Label(frame_count, text="Jumlah gambar variasi yang dihasilkan per 1 gambar asli:").pack()
        self.entry_aug_count = tk.Entry(frame_count, width=10, justify="center")
        self.entry_aug_count.insert(0, "3")
        self.entry_aug_count.pack(pady=5)

        tk.Button(self.tab_aug, text="MULAI AUGMENTASI", bg="#2196F3", fg="white", font=("Arial", 10, "bold"), command=self.run_aug).pack(pady=10, fill="x", padx=50)

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
            messagebox.showerror("Error", "Tidak ada file label (.txt) di folder '2_Exported_AnyLabeling'.")
            return

        # Mengambil parameter dari UI
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

            if len(transforms) > 0 and len(keypoints) > 0:
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
        messagebox.showinfo("Sukses", f"Augmentasi selesai!\n{berhasil} gambar asli berhasil diproses menjadi {berhasil * num_aug} gambar variasi.")

    # ================= TAB 3: SPLIT & YAML =================
    def setup_tab_split(self):
        tk.Label(self.tab_split, text="Split Dataset & Generate YAML", font=("Arial", 12, "bold")).pack(pady=15)
        
        frame_ratio = tk.Frame(self.tab_split)
        frame_ratio.pack(pady=15)
        tk.Label(frame_ratio, text="Train (%):").pack(side="left"); self.e_tr = tk.Entry(frame_ratio, width=4); self.e_tr.insert(0, "80"); self.e_tr.pack(side="left", padx=5)
        tk.Label(frame_ratio, text="Val (%):").pack(side="left"); self.e_vl = tk.Entry(frame_ratio, width=4); self.e_vl.insert(0, "10"); self.e_vl.pack(side="left", padx=5)
        tk.Label(frame_ratio, text="Test (%):").pack(side="left"); self.e_ts = tk.Entry(frame_ratio, width=4); self.e_ts.insert(0, "10"); self.e_ts.pack(side="left", padx=5)

        tk.Button(self.tab_split, text="SPLIT & BUAT DATASET.YAML", bg="#ff9800", fg="white", font=("Arial", 10, "bold"), command=self.run_split).pack(pady=25, fill="x", padx=50)

    def run_split(self):
        if not self.check_project_ready(): return
        
        src_dir = os.path.join(self.project_dir.get(), '3_Augmented')
        out_dir = os.path.join(self.project_dir.get(), '4_YOLO_Ready')
        classes_file = os.path.join(self.project_dir.get(), "classes.txt")
        
        if not os.path.exists(os.path.join(src_dir, 'images')):
            messagebox.showerror("Error", "Belum ada data di folder '3_Augmented'.")
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
        
        tr_idx = int(total * (float(self.e_tr.get()) / 100))
        vl_idx = tr_idx + int(total * (float(self.e_vl.get()) / 100))
        
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
        messagebox.showinfo("Selesai", f"Dataset siap di folder '4_YOLO_Ready'!")

    # ================= TAB 4: TRAINING =================
    def setup_tab_train(self):
        tk.Label(self.tab_train, text="Parameter Training YOLOv8 OBB", font=("Arial", 12, "bold")).pack(pady=10)
        
        frame_params = tk.Frame(self.tab_train)
        frame_params.pack(pady=5)
        
        tk.Label(frame_params, text="Nama Model Bawaan:").grid(row=0, column=0, sticky="w", pady=5)
        self.entry_model = tk.Entry(frame_params, width=15)
        self.entry_model.insert(0, "yolov8n-obb.pt")
        self.entry_model.grid(row=0, column=1, padx=10)
        
        tk.Label(frame_params, text="Jumlah Epochs Maksimal:").grid(row=1, column=0, sticky="w", pady=5)
        self.entry_epochs = tk.Entry(frame_params, width=15)
        self.entry_epochs.insert(0, "50")
        self.entry_epochs.grid(row=1, column=1, padx=10)
        
        tk.Label(frame_params, text="Image Size (Resolusi):").grid(row=2, column=0, sticky="w", pady=5)
        self.entry_imgsz = tk.Entry(frame_params, width=15)
        self.entry_imgsz.insert(0, "416") 
        self.entry_imgsz.grid(row=2, column=1, padx=10)
        
        tk.Label(frame_params, text="Batch Size (Proses per iterasi):").grid(row=3, column=0, sticky="w", pady=5)
        self.entry_batch = tk.Entry(frame_params, width=15)
        self.entry_batch.insert(0, "2") 
        self.entry_batch.grid(row=3, column=1, padx=10)

        tk.Label(frame_params, text="Patience (Batas Early Stop):").grid(row=4, column=0, sticky="w", pady=5)
        self.entry_patience = tk.Entry(frame_params, width=15)
        self.entry_patience.insert(0, "15") 
        self.entry_patience.grid(row=4, column=1, padx=10)

        tk.Label(frame_params, text="Workers (CPU Threads):").grid(row=5, column=0, sticky="w", pady=5)
        self.entry_workers = tk.Entry(frame_params, width=15)
        self.entry_workers.insert(0, "1") 
        self.entry_workers.grid(row=5, column=1, padx=10)
        
        tk.Label(frame_params, text="Device (0 = GPU, cpu = CPU):").grid(row=6, column=0, sticky="w", pady=5)
        self.entry_device = tk.Entry(frame_params, width=15)
        self.entry_device.insert(0, "0")
        self.entry_device.grid(row=6, column=1, padx=10)

        self.btn_train = tk.Button(self.tab_train, text="MULAI TRAINING", bg="#E91E63", fg="white", font=("Arial", 10, "bold"), command=self.start_train_thread)
        self.btn_train.pack(pady=15, fill="x", padx=50)
        
        self.lbl_train_status = tk.Label(self.tab_train, text="Status: Menunggu parameter dikonfigurasi...", fg="blue", font=("Arial", 11, "bold"))
        self.lbl_train_status.pack(pady=10)

    def start_train_thread(self):
        if not self.check_project_ready(): return
        
        self.btn_train.config(state=tk.DISABLED, bg="gray")
        self.lbl_train_status.config(text="Status: Memulai inisiasi model...")
        
        threading.Thread(target=self._run_train, daemon=True).start()

    def _run_train(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.root.after(0, lambda: messagebox.showerror("Error", "Library 'ultralytics' belum terinstal!"))
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL, bg="#E91E63"))
            self.root.after(0, lambda: self.lbl_train_status.config(text="Status: Gagal memuat library."))
            return

        yaml_path = os.path.join(self.project_dir.get(), '4_YOLO_Ready', 'dataset.yaml')
        if not os.path.exists(yaml_path):
            self.root.after(0, lambda: messagebox.showerror("Error", "File dataset.yaml belum ada!"))
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL, bg="#E91E63"))
            return

        # Mengambil parameter inputan user secara real-time
        model_name = self.entry_model.get()
        epochs = int(self.entry_epochs.get())
        imgsz = int(self.entry_imgsz.get())
        batch = int(self.entry_batch.get())
        patience = int(self.entry_patience.get())
        workers = int(self.entry_workers.get())
        device = self.entry_device.get()

        try:
            model = YOLO(model_name)
            
            def on_epoch_end(trainer):
                ep = trainer.epoch + 1
                tot = trainer.epochs
                self.root.after(0, lambda: self.lbl_train_status.config(text=f"Status: Proses Epoch {ep} dari {tot}..."))

            model.add_callback("on_fit_epoch_end", on_epoch_end)
            
            model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch, device=device, patience=patience, workers=workers)
            
            self.root.after(0, lambda: self.lbl_train_status.config(text="Status: Training Selesai 100%! Model tersimpan di folder 'runs'."))
            self.root.after(0, lambda: messagebox.showinfo("Selesai", "Proses Training telah berhasil diselesaikan!"))
            
        except Exception as e:
            self.root.after(0, lambda: self.lbl_train_status.config(text="Status: Training terhenti / Error."))
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL, bg="#E91E63"))

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloDataPipelineApp(root)
    root.mainloop()
