import os
import shutil
import random
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import albumentations as A

class YoloDataPipelineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 OBB Dataset Pipeline Tool")
        self.root.geometry("600x650")
        self.root.resizable(False, False)

        self.project_dir = tk.StringVar()

        self.setup_header()
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.tab_grab = ttk.Frame(self.notebook)
        self.tab_aug = ttk.Frame(self.notebook)
        self.tab_split = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_grab, text=" 1. Grab Dataset ")
        self.notebook.add(self.tab_aug, text=" 2. Augmentasi OBB ")
        self.notebook.add(self.tab_split, text=" 3. Split & YAML ")

        self.setup_tab_grab()
        self.setup_tab_aug()
        self.setup_tab_split()

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
            messagebox.showinfo("Project Disetel", f"Struktur folder berhasil dibuat di:\n{folder}\n\nSilakan lanjut ke Tab 1.")

    def check_project_ready(self):
        if not self.project_dir.get():
            messagebox.showwarning("Peringatan", "Pilih Folder Project Utama terlebih dahulu di bagian atas!")
            return False
        return True

    # ================= TAB 1: GRAB DATASET =================
    def setup_tab_grab(self):
        tk.Label(self.tab_grab, text="Setup Kelas & Grab Gambar", font=("Arial", 12, "bold")).pack(pady=15)
        
        tk.Label(self.tab_grab, text="Daftar Nama Objek/Kelas (pisahkan koma):").pack(anchor="w", padx=50)
        tk.Label(self.tab_grab, text="Aplikasi akan otomatis membuat classes.txt dari isian ini", fg="gray", font=("Arial", 8)).pack(anchor="w", padx=50)
        self.entry_cls = tk.Entry(self.tab_grab, width=50)
        self.entry_cls.insert(0, "truk, mobil")
        self.entry_cls.pack(pady=5, padx=50)

        frame_input = tk.Frame(self.tab_grab)
        frame_input.pack(pady=15)
        tk.Label(frame_input, text="Sumber Kamera (0 = WebCam / Link RTSP):").pack(anchor="w")
        self.entry_cam = tk.Entry(frame_input, width=50)
        self.entry_cam.insert(0, "0")
        self.entry_cam.pack(pady=5)
        
        tk.Label(self.tab_grab, text="Tekan 'S' untuk memotret, 'Q' untuk keluar.", fg="gray").pack(pady=5)
        tk.Button(self.tab_grab, text="SIMPAN KELAS & BUKA KAMERA", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), command=self.run_grab).pack(pady=15, fill="x", padx=50)

    def run_grab(self):
        if not self.check_project_ready(): return
        
        # 1. Generate classes.txt otomatis di folder project utama
        classes_text = self.entry_cls.get()
        classes_list = [c.strip() for c in classes_text.split(",") if c.strip()]
        if not classes_list:
            messagebox.showwarning("Peringatan", "Harap isi minimal 1 nama kelas!")
            return
            
        classes_file = os.path.join(self.project_dir.get(), "classes.txt")
        with open(classes_file, "w") as f:
            f.write("\n".join(classes_list))
            
        # 2. Proses Buka Kamera
        source = self.entry_cam.get()
        source = int(source) if source.isdigit() else source
        
        save_dir = os.path.join(self.project_dir.get(), '1_Raw_Images')
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Kamera/RTSP tidak dapat diakses!")
            return

        count = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
        messagebox.showinfo("Info", f"File classes.txt berhasil dibuat!\n\nKamera akan terbuka, fokus ke jendela kamera.\nTekan S untuk Simpan, Q untuk Keluar.")
        
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
        tk.Label(self.tab_aug, text="Augmentasi YOLO OBB", font=("Arial", 12, "bold")).pack(pady=15)
        tk.Label(self.tab_aug, text="Pastikan gambar sudah dilabeli & diekspor format YOLO OBB\nke folder '2_Exported_AnyLabeling'.\n(Gunakan classes.txt di folder project saat export!)", justify="center", fg="red").pack(pady=5)
        
        frame_aug = tk.Frame(self.tab_aug)
        frame_aug.pack(pady=15)
        tk.Label(frame_aug, text="Jumlah variasi per gambar:").pack(side="left")
        self.entry_aug_count = tk.Entry(frame_aug, width=5)
        self.entry_aug_count.insert(0, "3")
        self.entry_aug_count.pack(side="left", padx=10)

        tk.Button(self.tab_aug, text="MULAI AUGMENTASI", bg="#2196F3", fg="white", font=("Arial", 10, "bold"), command=self.run_aug).pack(pady=20, fill="x", padx=50)

    def run_aug(self):
        if not self.check_project_ready(): return
        
        input_dir = os.path.join(self.project_dir.get(), '2_Exported_AnyLabeling')
        output_dir = os.path.join(self.project_dir.get(), '3_Augmented')
        src_img_dir = os.path.join(input_dir, 'images')
        src_lbl_dir = os.path.join(input_dir, 'labels')
        
        if not os.path.exists(src_img_dir) or not os.listdir(src_img_dir):
            messagebox.showerror("Error", "Folder '2_Exported_AnyLabeling/images' kosong!")
            return

        out_img_dir = os.path.join(output_dir, 'images')
        out_lbl_dir = os.path.join(output_dir, 'labels')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        try: num_aug = int(self.entry_aug_count.get())
        except ValueError: num_aug = 1

        berhasil = 0
        for img_name in os.listdir(src_img_dir):
            if not img_name.lower().endswith(('.jpg', '.png')): continue
            base_name = os.path.splitext(img_name)[0]
            
            shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(out_img_dir, img_name))
            if os.path.exists(os.path.join(src_lbl_dir, base_name + '.txt')):
                shutil.copy(os.path.join(src_lbl_dir, base_name + '.txt'), os.path.join(out_lbl_dir, base_name + '.txt'))
            
            for i in range(num_aug):
                new_img_name = f"{base_name}_aug{i}.jpg"
                new_lbl_name = f"{base_name}_aug{i}.txt"
                shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(out_img_dir, new_img_name))
                if os.path.exists(os.path.join(src_lbl_dir, base_name + '.txt')):
                    shutil.copy(os.path.join(src_lbl_dir, base_name + '.txt'), os.path.join(out_lbl_dir, new_lbl_name))
            berhasil += 1

        messagebox.showinfo("Sukses", f"Augmentasi selesai!\n{berhasil} gambar berhasil dilipatgandakan.")

    # ================= TAB 3: SPLIT & YAML =================
    def setup_tab_split(self):
        tk.Label(self.tab_split, text="Split Dataset & Generate YAML", font=("Arial", 12, "bold")).pack(pady=15)
        
        frame_ratio = tk.Frame(self.tab_split)
        frame_ratio.pack(pady=15)
        tk.Label(frame_ratio, text="Train (%):").pack(side="left"); self.e_tr = tk.Entry(frame_ratio, width=4); self.e_tr.insert(0, "80"); self.e_tr.pack(side="left", padx=5)
        tk.Label(frame_ratio, text="Val (%):").pack(side="left"); self.e_vl = tk.Entry(frame_ratio, width=4); self.e_vl.insert(0, "10"); self.e_vl.pack(side="left", padx=5)
        tk.Label(frame_ratio, text="Test (%):").pack(side="left"); self.e_ts = tk.Entry(frame_ratio, width=4); self.e_ts.insert(0, "10"); self.e_ts.pack(side="left", padx=5)

        tk.Label(self.tab_split, text="Nama kelas akan dibaca otomatis dari\nfile classes.txt di folder project.", fg="green").pack(pady=10)
        tk.Button(self.tab_split, text="SPLIT & BUAT DATASET.YAML", bg="#ff9800", fg="white", font=("Arial", 10, "bold"), command=self.run_split).pack(pady=25, fill="x", padx=50)

    def run_split(self):
        if not self.check_project_ready(): return
        
        src_dir = os.path.join(self.project_dir.get(), '3_Augmented')
        out_dir = os.path.join(self.project_dir.get(), '4_YOLO_Ready')
        classes_file = os.path.join(self.project_dir.get(), "classes.txt")
        
        if not os.path.exists(os.path.join(src_dir, 'images')):
            messagebox.showerror("Error", "Belum ada data di folder '3_Augmented'.")
            return
            
        if not os.path.exists(classes_file):
            messagebox.showerror("Error", "File classes.txt tidak ditemukan di folder project!")
            return

        # Baca kelas dari file
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
        messagebox.showinfo("Selesai", f"Dataset siap di folder '4_YOLO_Ready'!\nFile dataset.yaml telah dibuat menggunakan kelas dari classes.txt.")

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloDataPipelineApp(root)
    root.mainloop()
