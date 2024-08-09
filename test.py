import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import NN_models_multi as NN_mm

torch.set_default_dtype(torch.float64)

def closure():
    optimizer.zero_grad()
    out = model(xt)
    loss = NN_mm.loss_mse(out, yt)
    print(loss)
    loss.backward()
    return loss


x = np.linspace(0, 1, 500)
xt = torch.from_numpy(x).reshape(-1,1)
y = np.sin(2*math.pi*x) + 0.2*np.cos(2*math.pi*x/0.1)
yt = torch.from_numpy(y).reshape(-1,1)

model = NN_mm.test_FBF_layer(22)

optimizer = torch.optim.LBFGS(model.parameters())

epochs = 500
print("Start")

for i in range(epochs):
    optimizer.step(closure)

print("Finish")

with torch.no_grad():
    pred = model(xt)

plt.plot(x, y, 'r')
plt.plot(x, pred.numpy(), 'b')

plt.show()
        
