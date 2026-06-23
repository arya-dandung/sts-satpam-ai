import os
import shutil
import random
import tkinter as tk
from tkinter import filedialog, messagebox

def jalankan_split():
    input_dir = entry_input.get()
    output_dir = entry_output.get()
    
    if not input_dir or not output_dir:
        messagebox.showwarning("Peringatan", "Harap pilih folder input dan output terlebih dahulu!")
        return

    src_images = os.path.join(input_dir, "images")
    src_labels = os.path.join(input_dir, "labels")

    if not os.path.exists(src_images) or not os.path.exists(src_labels):
        messagebox.showerror("Error", "Folder input harus memiliki sub-folder 'images' dan 'labels'.")
        return

    try:
        r_train = float(entry_train.get())
        r_val = float(entry_val.get())
        r_test = float(entry_test.get())
    except ValueError:
        messagebox.showerror("Error", "Rasio harus berupa angka!")
        return

    if r_train + r_val + r_test != 100:
        messagebox.showerror("Error", "Total persentase Train, Val, dan Test harus tepat 100%!")
        return

    classes_text = entry_classes.get().strip()
    if not classes_text:
        messagebox.showwarning("Peringatan", "Harap isi nama-nama objek/kelas!")
        return
    
    # Ambil daftar kelas dan bersihkan spasi
    classes_list = [c.strip() for c in classes_text.split(",")]

    # 1. Setup Folder Output
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 2. Acak Data
    all_files = [os.path.splitext(f)[0] for f in os.listdir(src_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(all_files)

    # 3. Hitung Pembagian
    total_data = len(all_files)
    train_idx = int(total_data * (r_train / 100))
    val_idx = train_idx + int(total_data * (r_val / 100))

    pembagian = {
        'train': all_files[:train_idx],
        'val': all_files[train_idx:val_idx],
        'test': all_files[val_idx:]
    }

    # 4. Copy File
    berhasil = 0
    for split_name, files in pembagian.items():
        for filename in files:
            for ext in ['.jpg', '.png', '.jpeg']:
                img_path = os.path.join(src_images, filename + ext)
                if os.path.exists(img_path):
                    shutil.copy(img_path, os.path.join(output_dir, 'images', split_name, filename + ext))
                    break
            
            lbl_path = os.path.join(src_labels, filename + '.txt')
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(output_dir, 'labels', split_name, filename + '.txt'))
            berhasil += 1

    # 5. Generate dataset.yaml
    yaml_content = f"path: {output_dir.replace(chr(92), '/')}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
    for i, c in enumerate(classes_list):
        yaml_content += f"  {i}: {c}\n"

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    messagebox.showinfo("Sukses", f"Proses Selesai!\nBerhasil membagi {berhasil} data.\nFile dataset.yaml juga telah dibuat otomatis.")

# --- Setup UI Tkinter ---
root = tk.Tk()
root.title("YOLO Dataset Splitter Tool")
root.geometry("450x500")
root.resizable(False, False)

def pilih_input(): entry_input.delete(0, tk.END); entry_input.insert(0, filedialog.askdirectory())
def pilih_output(): entry_output.delete(0, tk.END); entry_output.insert(0, filedialog.askdirectory())

# Elemen Folder
tk.Label(root, text="Folder Input (Dataset Gabungan):").pack(pady=(15,0), anchor="w", padx=20)
frame_in = tk.Frame(root); frame_in.pack(fill="x", padx=20)
entry_input = tk.Entry(frame_in, width=40); entry_input.pack(side="left", padx=(0,10))
tk.Button(frame_in, text="Pilih", command=pilih_input).pack(side="left")

tk.Label(root, text="Folder Output (Untuk hasil split):").pack(pady=(10,0), anchor="w", padx=20)
frame_out = tk.Frame(root); frame_out.pack(fill="x", padx=20)
entry_output = tk.Entry(frame_out, width=40); entry_output.pack(side="left", padx=(0,10))
tk.Button(frame_out, text="Pilih", command=pilih_output).pack(side="left")

# Elemen Rasio
tk.Label(root, text="Rasio Pembagian Dataset (%):").pack(pady=(20,0), anchor="w", padx=20)
frame_ratio = tk.Frame(root); frame_ratio.pack(fill="x", padx=20, pady=5)

tk.Label(frame_ratio, text="Train:").pack(side="left")
entry_train = tk.Entry(frame_ratio, width=5); entry_train.insert(0, "80"); entry_train.pack(side="left", padx=(5,15))

tk.Label(frame_ratio, text="Val:").pack(side="left")
entry_val = tk.Entry(frame_ratio, width=5); entry_val.insert(0, "10"); entry_val.pack(side="left", padx=(5,15))

tk.Label(frame_ratio, text="Test:").pack(side="left")
entry_test = tk.Entry(frame_ratio, width=5); entry_test.insert(0, "10"); entry_test.pack(side="left", padx=(5,0))

# Elemen Kelas
tk.Label(root, text="Daftar Nama Objek (pisahkan dengan koma):").pack(pady=(20,0), anchor="w", padx=20)
tk.Label(root, text="Contoh: truk, mobil, motor", fg="gray", font=("Arial", 8)).pack(anchor="w", padx=20)
entry_classes = tk.Entry(root, width=55); entry_classes.pack(padx=20, pady=5)

# Tombol Eksekusi
tk.Button(root, text="MULAI SPLIT DATASET", bg="#008CBA", fg="white", font=("Arial", 10, "bold"), command=jalankan_split).pack(pady=25, fill="x", padx=20)

root.mainloop()
