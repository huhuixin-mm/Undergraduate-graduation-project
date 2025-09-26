import torch, os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.version.cuda    =", torch.version.cuda)
print("torch.cuda.is_available() =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count =", torch.cuda.device_count())
    print("device name  =", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("ok, tensor on:", x.device)

