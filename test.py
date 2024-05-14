from kan import KAN
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# without cuda
start = time.time()
model = KAN(width=[768,64,2], grid=5, k=3)
x = torch.normal(0,0.5,size=(4,768))
y = model(x)
end = time.time()
print(end - start) # 3.04 s

# with cuda
start = time.time()
model = KAN(width=[768,64,2], grid=5, k=3, device = device)
x = torch.normal(0,0.5,size=(4,768)).to(device)
y = model(x)
end = time.time()
print(end - start) # 10.9s
