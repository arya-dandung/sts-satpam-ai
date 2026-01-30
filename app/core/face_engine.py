# #Face Rec GPU

# import cv2
# import numpy as np
# import os
# import onnxruntime
# from insightface.app import FaceAnalysis

# class FaceEngine:
#     def __init__(self, folder_path="known_faces"):
#         print("🧠 Mencoba memuat InsightFace ke GPU (GTX 745)...")
        
#         # Cek apakah ONNX Runtime mendeteksi CUDA
#         available_providers = onnxruntime.get_available_providers()
#         print(f"ℹ️  ONNX Providers: {available_providers}")

#         try:
#             if 'CUDAExecutionProvider' in available_providers:
#                 # --- SETTING GPU ---
#                 # ctx_id=0 artinya GPU index 0
#                 self.app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider'])
#                 self.app.prepare(ctx_id=0, det_size=(640, 640))
#                 print("🚀 SUKSES: Face Engine berjalan di GPU!")
#             else:
#                 raise Exception("CUDA Provider tidak ditemukan di ONNX Runtime")
                
#         except Exception as e:
#             print(f"⚠️ GAGAL GPU ({e}). Beralih ke CPU...")
#             # --- FALLBACK CPU ---
#             # ctx_id=-1 artinya CPU
#             self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
#             self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
#         # List data wajah
#         self.known_embeddings = []
#         self.known_names = []
#         self.similarity_threshold = 0.5
        
#         self.load_database(folder_path)

#     def load_database(self, root_path):
#         if not os.path.exists(root_path):
#             os.makedirs(root_path)
#             return

#         print(f"📂 Membaca Database dari: {root_path}")
#         total_images = 0
        
#         for item_name in os.listdir(root_path):
#             item_path = os.path.join(root_path, item_name)
            
#             if os.path.isdir(item_path):
#                 person_name = item_name
#                 for filename in os.listdir(item_path):
#                     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                         self._process_image(os.path.join(item_path, filename), person_name)
#                         total_images += 1
            
#             elif item_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 person_name = os.path.splitext(item_name)[0]
#                 self._process_image(item_path, person_name)
#                 total_images += 1
        
#         unique_people = len(set(self.known_names))
#         print(f"✨ Database Siap: {unique_people} Orang ({total_images} sampel).")

#     def _process_image(self, image_path, name):
#         img = cv2.imread(image_path)
#         if img is None: return

#         # InsightFace akan otomatis menggunakan GPU/CPU sesuai inisialisasi di atas
#         faces = self.app.get(img)
        
#         if len(faces) > 0:
#             self.known_embeddings.append(faces[0].embedding)
#             self.known_names.append(name)
#         else:
#             print(f"   ⚠️ Wajah tidak terdeteksi di: {image_path}")

#     def recognize_crop(self, face_img_bgr):
#         # Proses ini sekarang akan memakan VRAM, bukan RAM
#         faces = self.app.get(face_img_bgr)
        
#         if len(faces) == 0:
#             return "Orang Asing"
        
#         target_emb = faces[0].embedding
#         max_score = 0
#         best_name = "Orang Asing"

#         for idx, known_emb in enumerate(self.known_embeddings):
#             score = np.dot(target_emb, known_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(known_emb))
            
#             if score > max_score:
#                 max_score = score
#                 if score > self.similarity_threshold:
#                     best_name = self.known_names[idx]
        
#         return best_name

# #=======================================================================
#Face Rec CPU
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, folder_path="known_faces"):
        print("🧠 [CPU] Memuat Model Wajah (InsightFace Multi-Sample)...")
        
        # Mode CPU (ctx_id = -1)
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        # List data wajah
        self.known_embeddings = []
        self.known_names = []
        
        # Threshold Kemiripan (Makin tinggi makin ketat)
        # Karena kita punya banyak sampel, kita bisa sedikit lebih ketat (0.5 - 0.6)
        self.similarity_threshold = 0.4
        
        self.load_database(folder_path)

    def load_database(self, root_path):
        """
        Membaca database wajah dengan dukungan MULTIPLE IMAGES per orang.
        Mendukung struktur folder: known_faces/NamaOrang/foto1.jpg
        """
        if not os.path.exists(root_path):
            os.makedirs(root_path)
            print(f"⚠️ Folder '{root_path}' dibuat. Silakan isi dengan folder nama orang.")
            return

        print(f"📂 Membaca Database dari: {root_path}")
        
        total_images = 0
        
        # Loop semua item di dalam folder known_faces
        for item_name in os.listdir(root_path):
            item_path = os.path.join(root_path, item_name)
            
            # 1. Jika item adalah FOLDER (Contoh: known_faces/Arya/)
            if os.path.isdir(item_path):
                person_name = item_name  # Nama orang = Nama folder
                
                # Baca semua foto di dalam folder tersebut
                for filename in os.listdir(item_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(item_path, filename)
                        self._process_image(file_path, person_name)
                        total_images += 1
            
            # 2. Jika item adalah FILE (Support legacy: known_faces/arya.jpg)
            elif item_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                person_name = os.path.splitext(item_name)[0]
                self._process_image(item_path, person_name)
                total_images += 1
        
        # Hitung statistik
        unique_people = len(set(self.known_names))
        print(f"✨ Database Siap: {unique_people} Orang dari {total_images} Foto Referensi.")

    def _process_image(self, image_path, name):
        """Helper function untuk memproses satu gambar"""
        img = cv2.imread(image_path)
        if img is None: return

        # Deteksi wajah di foto referensi
        faces = self.app.get(img)
        
        if len(faces) > 0:
            # Ambil wajah terbesar (asumsi foto profil satu orang)
            # Simpan embedding dan namanya
            self.known_embeddings.append(faces[0].embedding)
            self.known_names.append(name)
            # print(f"   ✅ Loaded: {name} ({os.path.basename(image_path)})")
        else:
            print(f"   ⚠️ Wajah tidak terdeteksi di: {image_path}")

    def recognize_crop(self, face_img_bgr):
        """
        Mengenali wajah.
        Logika: Mencari kemiripan tertinggi dari SEMUA sampel yang ada.
        """
        faces = self.app.get(face_img_bgr)
        
        if len(faces) == 0:
            return "Orang Asing"
        
        target_emb = faces[0].embedding
        max_score = 0
        best_name = "Orang Asing"

        # Bandingkan wajah target dengan RATUSAN sampel di database
        # InsightFace menggunakan Dot Product untuk Cosine Similarity
        # (Karena embedding biasanya sudah dinormalisasi)
        
        for idx, known_emb in enumerate(self.known_embeddings):
            # Rumus Cosine Similarity
            score = np.dot(target_emb, known_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(known_emb))
            
            if score > max_score:
                max_score = score
                if score > self.similarity_threshold:
                    best_name = self.known_names[idx]
        
        return best_name