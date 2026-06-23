import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import albumentations as A

def jalankan_augmentasi():
    input_dir = entry_input.get()
    output_dir = entry_output.get()
    
    if not input_dir or not output_dir:
        messagebox.showwarning("Peringatan", "Harap pilih folder input dan output terlebih dahulu!")
        return

    input_img_dir = os.path.join(input_dir, "images")
    input_lbl_dir = os.path.join(input_dir, "labels")
    output_img_dir = os.path.join(output_dir, "images")
    output_lbl_dir = os.path.join(output_dir, "labels")

    if not os.path.exists(input_img_dir) or not os.path.exists(input_lbl_dir):
        messagebox.showerror("Error", "Folder input harus memiliki sub-folder 'images' dan 'labels'.")
        return

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    augmentations = []
    if var_flip_h.get(): augmentations.append(A.HorizontalFlip(p=0.5))
    if var_flip_v.get(): augmentations.append(A.VerticalFlip(p=0.5))
    if var_rotate.get(): augmentations.append(A.Rotate(limit=45, p=0.7))
    if var_bright.get(): augmentations.append(A.RandomBrightnessContrast(p=0.5))
    
    if not augmentations:
        messagebox.showwarning("Peringatan", "Pilih minimal 1 jenis augmentasi!")
        return

    transform = A.Compose(augmentations, keypoint_params=A.KeypointParams(format='xy'))
    jumlah_variasi = int(entry_variasi.get())
    berhasil = 0
    
    for label_file in os.listdir(input_lbl_dir):
        if not label_file.endswith(".txt"): continue
        base_name = os.path.splitext(label_file)[0]
        img_path = os.path.join(input_img_dir, f"{base_name}.jpg")
        lbl_path = os.path.join(input_lbl_dir, label_file)

        if not os.path.exists(img_path): continue

        image = cv2.imread(img_path)
        if image is None: continue
        h, w, _ = image.shape

        with open(lbl_path, "r") as f: lines = f.readlines()

        keypoints, class_ids = [], []
        for line in lines:
            parts = line.strip().split()
            class_ids.append(int(parts[0]))
            pts = [float(x) for x in parts[1:]]
            keypoints.extend([(pts[0]*w, pts[1]*h), (pts[2]*w, pts[3]*h), (pts[4]*w, pts[5]*h), (pts[6]*w, pts[7]*h)])

        for i in range(jumlah_variasi):
            try:
                transformed = transform(image=image, keypoints=keypoints)
                aug_img, aug_kps = transformed['image'], transformed['keypoints']
                aug_img_name = f"{base_name}_aug{i+1}.jpg"
                aug_lbl_name = f"{base_name}_aug{i+1}.txt"

                cv2.imwrite(os.path.join(output_img_dir, aug_img_name), aug_img)

                with open(os.path.join(output_lbl_dir, aug_lbl_name), "w") as f:
                    for j in range(len(class_ids)):
                        idx = j * 4
                        if len(aug_kps) > idx + 3:
                            x1, y1 = aug_kps[idx][0]/w, aug_kps[idx][1]/h
                            x2, y2 = aug_kps[idx+1][0]/w, aug_kps[idx+1][1]/h
                            x3, y3 = aug_kps[idx+2][0]/w, aug_kps[idx+2][1]/h
                            x4, y4 = aug_kps[idx+3][0]/w, aug_kps[idx+3][1]/h
                            pts_norm = [max(0.0, min(1.0, val)) for val in [x1, y1, x2, y2, x3, y3, x4, y4]]
                            f.write(f"{class_ids[j]} " + " ".join([f"{val:.6f}" for val in pts_norm]) + "\n")
                berhasil += 1
            except Exception as e:
                print(f"Error: {e}")

    messagebox.showinfo("Sukses", f"Proses selesai! Menghasilkan {berhasil} file augmentasi.")

root = tk.Tk()
root.title("YOLO OBB Augmentation Tool")
root.geometry("450x400")
root.resizable(False, False)

def pilih_input(): entry_input.delete(0, tk.END); entry_input.insert(0, filedialog.askdirectory())
def pilih_output(): entry_output.delete(0, tk.END); entry_output.insert(0, filedialog.askdirectory())

tk.Label(root, text="Folder Input (Harus berisi 'images' & 'labels'):").pack(pady=(10,0), anchor="w", padx=20)
frame_in = tk.Frame(root); frame_in.pack(fill="x", padx=20)
entry_input = tk.Entry(frame_in, width=40); entry_input.pack(side="left", padx=(0,10))
tk.Button(frame_in, text="Pilih", command=pilih_input).pack(side="left")

tk.Label(root, text="Folder Output:").pack(pady=(10,0), anchor="w", padx=20)
frame_out = tk.Frame(root); frame_out.pack(fill="x", padx=20)
entry_output = tk.Entry(frame_out, width=40); entry_output.pack(side="left", padx=(0,10))
tk.Button(frame_out, text="Pilih", command=pilih_output).pack(side="left")

tk.Label(root, text="Jenis Augmentasi:").pack(pady=(15,0), anchor="w", padx=20)
var_flip_h = tk.BooleanVar(value=True); tk.Checkbutton(root, text="Horizontal Flip", variable=var_flip_h).pack(anchor="w", padx=20)
var_flip_v = tk.BooleanVar(); tk.Checkbutton(root, text="Vertical Flip", variable=var_flip_v).pack(anchor="w", padx=20)
var_rotate = tk.BooleanVar(value=True); tk.Checkbutton(root, text="Rotate (Acak s.d 45 derajat)", variable=var_rotate).pack(anchor="w", padx=20)
var_bright = tk.BooleanVar(value=True); tk.Checkbutton(root, text="Random Brightness & Contrast", variable=var_bright).pack(anchor="w", padx=20)

frame_var = tk.Frame(root); frame_var.pack(fill="x", padx=20, pady=15)
tk.Label(frame_var, text="Jumlah variasi per gambar:").pack(side="left")
entry_variasi = tk.Entry(frame_var, width=5); entry_variasi.insert(0, "3"); entry_variasi.pack(side="left", padx=10)

tk.Button(root, text="MULAI AUGMENTASI", bg="green", fg="white", font=("Arial", 10, "bold"), command=jalankan_augmentasi).pack(pady=10, fill="x", padx=20)
root.mainloop()
