import torch
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU bulundu: {torch.cuda.get_device_name(0)}")
    print("Cuda test ediliyor...")

    start_time = time.time()
    duration = 60  # saniye
    tensors = []

    while time.time() - start_time < duration:
        # 10.000 x 10.000 float32 tensör 400mb
        x = torch.rand((10000, 10000), device=device)
        tensors.append(x)  

        y = torch.mm(x, x)
        _ = y.sum().item() 

        if len(tensors) > 5:
            tensors.pop(0)
    
    print("Test tamamlandi.")
else:
    print("CUDA GPU bulunamadi, CPU kullaniliyor.")
