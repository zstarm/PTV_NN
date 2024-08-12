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


x = np.linspace(0, 1, 200)
t = np.linspace(0, 1, 200)

xt = torch.empty((200*200, 2))
yt = torch.empty(200*200,1)
for i in range(0,200):
    for j in range(0,200):
        xt[j+(i*200),0] = x[i]
        xt[j+(i*200),1] = t[j]
        yt[j+(i*200),0] = np.sin(2*math.pi*x[i] - 2*math.pi*t[j]) + 0.2*np.cos(2*math.pi*x[i]/0.1)


model = NN_mm.test_FBF_layer(22)

optimizer = torch.optim.LBFGS(model.parameters())

epochs = 75
print("Start")

for i in range(epochs):
    optimizer.step(closure)

print("Finish")

with torch.no_grad():
    pred = model(xt)

plt.plot(xt.numpy()[-200:,0], yt.numpy()[-200:,0], 'r')
plt.plot(xt.numpy()[-200:,0], pred.numpy()[-200:,0], 'b')

plt.show()
        
