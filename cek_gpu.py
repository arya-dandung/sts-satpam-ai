import torch
print(f"Versi PyTorch: {torch.version}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nama GPU: {torch.cuda.get_device_name(0)}")
    print(f"Versi CUDA: {torch.version.cuda}")
    # Tes kemampuan komputasi (GTX 745 harusnya 5.0)
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("❌ Masih terdeteksi CPU. GPU tidak kompatibel dengan versi ini.")