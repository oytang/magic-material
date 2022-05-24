
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# In[2]:


toy_alloy_data = pd.read_csv("../toy_alloy_data.csv")

X_col = [f'element {i}' for i in range(1,13)]
Y_col = [f'property {i}' for i in range(1,7)]

X = toy_alloy_data[X_col].values
Y = toy_alloy_data[Y_col].values

# scale Y
scaler = StandardScaler()
Y = scaler.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# move tensors to GPU (or CPU)
X_train = torch.tensor(X_train).to(device)
Y_train = torch.tensor(Y_train).to(device)
X_test = torch.tensor(X_test).to(device)
Y_test = torch.tensor(Y_test).to(device)


# In[3]:


model = nn.Sequential(
    nn.Linear(6,9),
    nn.Softmax(1),
    nn.Linear(9,16),
    nn.Softmax(1),
    nn.Linear(16,12),
    nn.Softmax(1)
)
model.to(device)
model.double()


# In[4]:


# a simple training procedure with MSE loss and Adam optimiser
# directly copied from example notebook
def train(m, x, y,  max_iter):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters())
    loss_hist = []
    R2_hist = []
    y_bar = torch.mean(y, 0)

    for t in range(1, max_iter + 1):
        y_pred = m(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 100 == 0:
            loss_hist.append(loss.detach().cpu())
            R2 = 1 - torch.sum((y - y_pred)**2, 0) / torch.sum((y - y_bar)**2, 0)
            R2 = torch.mean(R2)
            R2_hist.append(R2.detach().cpu())
            print(f'epoch: {t}, loss: {float(loss.item()):.4f}, R^2: {float(R2):.4f}')
            if len(loss_hist) > 2 and torch.abs((loss_hist[-1]-loss_hist[-2])/loss_hist[-1]) < 1e-4:
                break
    return m, loss_hist, R2_hist


# In[5]:


m, loss_hist, R2_hist= train(model, Y_train, X_train, max_iter=30000)


# In[6]:


plt.plot(loss_hist)


# In[7]:


plt.plot(R2_hist)


# In[8]:


with torch.no_grad():
    print(mean_absolute_error(model(Y_test).cpu().numpy(), X_test.cpu().numpy()))


# In[9]:


with torch.no_grad():
    X_pred = model(Y_test).cpu().numpy()
    X_true = X_test.cpu().numpy()

fig, axs = plt.subplots(4, 3, figsize=(12,12))
for i in range(12):
    X_bar = np.mean(X_true[:,i])
    R_2 = 1 - np.sum((X_true[:,i] - X_pred[:,i])**2) / np.sum((X_true[:,i] - X_bar)**2)
    ax = axs[i // 3, i % 3]
    ax.plot(X_true[:, i], X_pred[:, i], '.')
    ax.plot(X_true[:, i], X_true[:, i], '-')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('true')
    ax.set_ylabel('predicted')
    ax.set_title(f'{X_col[i]}, R^2={R_2:.4f}')

plt.tight_layout()
plt.show()

"""
epoch: 100, loss: 0.0566, R^2: -4905.3634
epoch: 200, loss: 0.0528, R^2: -3666.1058
epoch: 300, loss: 0.0489, R^2: -2654.4316
epoch: 400, loss: 0.0434, R^2: -1875.0973
epoch: 500, loss: 0.0359, R^2: -1229.9492
epoch: 600, loss: 0.0294, R^2: -756.0875
epoch: 700, loss: 0.0255, R^2: -474.2930
epoch: 800, loss: 0.0234, R^2: -318.3846
epoch: 900, loss: 0.0223, R^2: -230.5552
epoch: 1000, loss: 0.0217, R^2: -178.5043
epoch: 1100, loss: 0.0213, R^2: -145.8739
epoch: 1200, loss: 0.0210, R^2: -123.8470
epoch: 1300, loss: 0.0208, R^2: -108.5064
epoch: 1400, loss: 0.0206, R^2: -97.3818
epoch: 1500, loss: 0.0205, R^2: -89.2680
epoch: 1600, loss: 0.0203, R^2: -83.3794
epoch: 1700, loss: 0.0201, R^2: -79.4640
epoch: 1800, loss: 0.0198, R^2: -77.7215
epoch: 1900, loss: 0.0192, R^2: -78.6507
epoch: 2000, loss: 0.0178, R^2: -82.1537
epoch: 2100, loss: 0.0158, R^2: -73.7252
epoch: 2200, loss: 0.0134, R^2: -69.2823
epoch: 2300, loss: 0.0118, R^2: -60.1944
epoch: 2400, loss: 0.0108, R^2: -53.6649
epoch: 2500, loss: 0.0101, R^2: -49.4375
epoch: 2600, loss: 0.0095, R^2: -46.5942
epoch: 2700, loss: 0.0091, R^2: -44.7256
epoch: 2800, loss: 0.0087, R^2: -43.2587
epoch: 2900, loss: 0.0083, R^2: -40.7337
epoch: 3000, loss: 0.0079, R^2: -36.4444
epoch: 3100, loss: 0.0077, R^2: -31.5396
epoch: 3200, loss: 0.0074, R^2: -27.0755
epoch: 3300, loss: 0.0073, R^2: -23.3228
epoch: 3400, loss: 0.0071, R^2: -20.2055
epoch: 3500, loss: 0.0070, R^2: -17.6234
epoch: 3600, loss: 0.0068, R^2: -15.5091
epoch: 3700, loss: 0.0067, R^2: -13.7887
epoch: 3800, loss: 0.0066, R^2: -12.3758
epoch: 3900, loss: 0.0065, R^2: -11.1961
epoch: 4000, loss: 0.0065, R^2: -10.1933
epoch: 4100, loss: 0.0064, R^2: -9.3309
epoch: 4200, loss: 0.0064, R^2: -8.5860
epoch: 4300, loss: 0.0063, R^2: -7.9405
epoch: 4400, loss: 0.0063, R^2: -7.3786
epoch: 4500, loss: 0.0062, R^2: -6.8833
epoch: 4600, loss: 0.0062, R^2: -6.4345
epoch: 4700, loss: 0.0062, R^2: -6.0480
epoch: 4800, loss: 0.0061, R^2: -5.6848
epoch: 4900, loss: 0.0061, R^2: -5.3631
epoch: 5000, loss: 0.0061, R^2: -5.0629
epoch: 5100, loss: 0.0060, R^2: -4.7821
epoch: 5200, loss: 0.0060, R^2: -4.5202
epoch: 5300, loss: 0.0060, R^2: -4.2771
epoch: 5400, loss: 0.0060, R^2: -4.0522
epoch: 5500, loss: 0.0059, R^2: -3.8445
epoch: 5600, loss: 0.0059, R^2: -3.6535
epoch: 5700, loss: 0.0059, R^2: -3.4799
epoch: 5800, loss: 0.0059, R^2: -3.3263
epoch: 5900, loss: 0.0059, R^2: -3.2013
epoch: 6000, loss: 0.0058, R^2: -3.1540
epoch: 6100, loss: 0.0057, R^2: -3.1630
epoch: 6200, loss: 0.0056, R^2: -3.0602
epoch: 6300, loss: 0.0056, R^2: -2.8706
epoch: 6400, loss: 0.0055, R^2: -2.6502
epoch: 6500, loss: 0.0055, R^2: -2.4451
epoch: 6600, loss: 0.0055, R^2: -2.2592
epoch: 6700, loss: 0.0055, R^2: -2.0922
epoch: 6800, loss: 0.0054, R^2: -1.9428
epoch: 6900, loss: 0.0054, R^2: -1.8086
epoch: 7000, loss: 0.0054, R^2: -1.6874
epoch: 7100, loss: 0.0054, R^2: -1.5775
epoch: 7200, loss: 0.0054, R^2: -1.4775
epoch: 7300, loss: 0.0053, R^2: -1.3916
epoch: 7400, loss: 0.0053, R^2: -1.3151
epoch: 7500, loss: 0.0053, R^2: -1.2207
epoch: 7600, loss: 0.0052, R^2: -1.1274
epoch: 7700, loss: 0.0052, R^2: -1.0404
epoch: 7800, loss: 0.0052, R^2: -0.9535
epoch: 7900, loss: 0.0051, R^2: -0.8712
epoch: 8000, loss: 0.0051, R^2: -0.7888
epoch: 8100, loss: 0.0050, R^2: -0.7142
epoch: 8200, loss: 0.0050, R^2: -0.6452
epoch: 8300, loss: 0.0050, R^2: -0.5835
epoch: 8400, loss: 0.0050, R^2: -0.5294
epoch: 8500, loss: 0.0050, R^2: -0.4819
epoch: 8600, loss: 0.0049, R^2: -0.4392
epoch: 8700, loss: 0.0049, R^2: -0.4001
epoch: 8800, loss: 0.0049, R^2: -0.3636
epoch: 8900, loss: 0.0049, R^2: -0.3287
epoch: 9000, loss: 0.0049, R^2: -0.2946
epoch: 9100, loss: 0.0049, R^2: -0.2599
epoch: 9200, loss: 0.0049, R^2: -0.2263
epoch: 9300, loss: 0.0049, R^2: -0.1920
epoch: 9400, loss: 0.0049, R^2: -0.1581
epoch: 9500, loss: 0.0049, R^2: -0.1252
epoch: 9600, loss: 0.0048, R^2: -0.0937
epoch: 9700, loss: 0.0048, R^2: -0.0638
epoch: 9800, loss: 0.0048, R^2: -0.0353
epoch: 9900, loss: 0.0048, R^2: -0.0087
epoch: 10000, loss: 0.0048, R^2: 0.0168
epoch: 10100, loss: 0.0048, R^2: 0.0401
epoch: 10200, loss: 0.0048, R^2: 0.0615
epoch: 10300, loss: 0.0048, R^2: 0.0811
epoch: 10400, loss: 0.0048, R^2: 0.0997
epoch: 10500, loss: 0.0048, R^2: 0.1168
epoch: 10600, loss: 0.0048, R^2: 0.1328
epoch: 10700, loss: 0.0048, R^2: 0.1479
epoch: 10800, loss: 0.0048, R^2: 0.1619
epoch: 10900, loss: 0.0048, R^2: 0.1753
epoch: 11000, loss: 0.0047, R^2: 0.1878
epoch: 11100, loss: 0.0047, R^2: 0.1996
epoch: 11200, loss: 0.0047, R^2: 0.2104
epoch: 11300, loss: 0.0047, R^2: 0.2211
epoch: 11400, loss: 0.0047, R^2: 0.2328
epoch: 11500, loss: 0.0047, R^2: 0.2439
epoch: 11600, loss: 0.0047, R^2: 0.2548
epoch: 11700, loss: 0.0047, R^2: 0.2652
epoch: 11800, loss: 0.0047, R^2: 0.2747
epoch: 11900, loss: 0.0047, R^2: 0.2840
epoch: 12000, loss: 0.0047, R^2: 0.2918
epoch: 12100, loss: 0.0047, R^2: 0.2965
epoch: 12200, loss: 0.0047, R^2: 0.3008
epoch: 12300, loss: 0.0047, R^2: 0.3046
epoch: 12400, loss: 0.0047, R^2: 0.3075
epoch: 12500, loss: 0.0046, R^2: 0.3097
epoch: 12600, loss: 0.0046, R^2: 0.3112
epoch: 12700, loss: 0.0046, R^2: 0.3129
epoch: 12800, loss: 0.0046, R^2: 0.3160
epoch: 12900, loss: 0.0046, R^2: 0.3189
epoch: 13000, loss: 0.0046, R^2: 0.3215
epoch: 13100, loss: 0.0046, R^2: 0.3242
epoch: 13200, loss: 0.0046, R^2: 0.3267
epoch: 13300, loss: 0.0045, R^2: 0.3290
epoch: 13400, loss: 0.0045, R^2: 0.3287
epoch: 13500, loss: 0.0045, R^2: 0.3275
epoch: 13600, loss: 0.0045, R^2: 0.3251
epoch: 13700, loss: 0.0045, R^2: 0.3249
epoch: 13800, loss: 0.0045, R^2: 0.3282
epoch: 13900, loss: 0.0045, R^2: 0.3326
epoch: 14000, loss: 0.0045, R^2: 0.3372
epoch: 14100, loss: 0.0045, R^2: 0.3415
epoch: 14200, loss: 0.0045, R^2: 0.3454
epoch: 14300, loss: 0.0045, R^2: 0.3489
epoch: 14400, loss: 0.0045, R^2: 0.3521
epoch: 14500, loss: 0.0045, R^2: 0.3548
epoch: 14600, loss: 0.0045, R^2: 0.3572
epoch: 14700, loss: 0.0045, R^2: 0.3589
epoch: 14800, loss: 0.0045, R^2: 0.3598
epoch: 14900, loss: 0.0044, R^2: 0.3596
epoch: 15000, loss: 0.0044, R^2: 0.3571
epoch: 15100, loss: 0.0044, R^2: 0.3512
epoch: 15200, loss: 0.0044, R^2: 0.3393
epoch: 15300, loss: 0.0044, R^2: 0.3176
epoch: 15400, loss: 0.0044, R^2: 0.2813
epoch: 15500, loss: 0.0044, R^2: 0.2273
epoch: 15600, loss: 0.0044, R^2: 0.1571
epoch: 15700, loss: 0.0044, R^2: 0.0780
epoch: 15800, loss: 0.0044, R^2: -0.0054
epoch: 15900, loss: 0.0044, R^2: -0.0945
epoch: 16000, loss: 0.0044, R^2: -0.1975
epoch: 16100, loss: 0.0044, R^2: -0.3259
epoch: 16200, loss: 0.0044, R^2: -0.4887
epoch: 16300, loss: 0.0044, R^2: -0.6879
epoch: 16400, loss: 0.0044, R^2: -0.9042
epoch: 16500, loss: 0.0044, R^2: -1.1132
epoch: 16600, loss: 0.0044, R^2: -1.2990
epoch: 16700, loss: 0.0044, R^2: -1.4521
epoch: 16800, loss: 0.0044, R^2: -1.5813
epoch: 16900, loss: 0.0044, R^2: -1.6950
epoch: 17000, loss: 0.0044, R^2: -1.7997
epoch: 17100, loss: 0.0044, R^2: -1.8980
epoch: 17200, loss: 0.0044, R^2: -1.9841
epoch: 17300, loss: 0.0044, R^2: -2.0601
epoch: 17400, loss: 0.0044, R^2: -2.1485
epoch: 17500, loss: 0.0044, R^2: -2.2628
epoch: 17600, loss: 0.0044, R^2: -2.4164
epoch: 17700, loss: 0.0044, R^2: -2.6258
epoch: 17800, loss: 0.0044, R^2: -2.9109
epoch: 17900, loss: 0.0044, R^2: -3.2800
epoch: 18000, loss: 0.0044, R^2: -3.7185
epoch: 18100, loss: 0.0044, R^2: -4.1802
epoch: 18200, loss: 0.0044, R^2: -4.6134
epoch: 18300, loss: 0.0044, R^2: -4.9893
epoch: 18400, loss: 0.0044, R^2: -5.3086
epoch: 18500, loss: 0.0044, R^2: -5.5833
epoch: 18600, loss: 0.0043, R^2: -5.8208
epoch: 18700, loss: 0.0043, R^2: -5.9894
epoch: 18800, loss: 0.0043, R^2: -6.1918
epoch: 18900, loss: 0.0043, R^2: -6.4061
epoch: 19000, loss: 0.0043, R^2: -6.6124
epoch: 19100, loss: 0.0043, R^2: -6.8220
epoch: 19200, loss: 0.0043, R^2: -6.9942
epoch: 19300, loss: 0.0043, R^2: -7.1726
epoch: 19400, loss: 0.0043, R^2: -7.3254
epoch: 19500, loss: 0.0043, R^2: -7.4694
epoch: 19600, loss: 0.0043, R^2: -7.5806
epoch: 19700, loss: 0.0043, R^2: -7.6550
epoch: 19800, loss: 0.0043, R^2: -7.7726
epoch: 19900, loss: 0.0043, R^2: -7.8430
epoch: 20000, loss: 0.0043, R^2: -7.8892
epoch: 20100, loss: 0.0043, R^2: -7.9218
epoch: 20200, loss: 0.0043, R^2: -7.9529
epoch: 20300, loss: 0.0043, R^2: -7.9794
epoch: 20400, loss: 0.0043, R^2: -8.0133
epoch: 20500, loss: 0.0043, R^2: -8.0506
epoch: 20600, loss: 0.0043, R^2: -8.0991
epoch: 20700, loss: 0.0043, R^2: -8.1393
epoch: 20800, loss: 0.0043, R^2: -8.2068
epoch: 20900, loss: 0.0043, R^2: -8.2687
epoch: 21000, loss: 0.0043, R^2: -8.2951
epoch: 21100, loss: 0.0043, R^2: -8.3868
epoch: 21200, loss: 0.0043, R^2: -8.4423
epoch: 21300, loss: 0.0043, R^2: -8.4973
epoch: 21400, loss: 0.0043, R^2: -8.5515
epoch: 21500, loss: 0.0043, R^2: -8.6087
epoch: 21600, loss: 0.0043, R^2: -8.6557
epoch: 21700, loss: 0.0043, R^2: -8.6908
epoch: 21800, loss: 0.0043, R^2: -8.7380
epoch: 21900, loss: 0.0043, R^2: -8.7719
epoch: 22000, loss: 0.0043, R^2: -8.8032
epoch: 22100, loss: 0.0043, R^2: -8.8478
epoch: 22200, loss: 0.0043, R^2: -8.8798
epoch: 22300, loss: 0.0043, R^2: -8.9119
epoch: 22400, loss: 0.0043, R^2: -8.9443
epoch: 22500, loss: 0.0043, R^2: -8.9714
epoch: 22600, loss: 0.0043, R^2: -9.0000
epoch: 22700, loss: 0.0043, R^2: -9.0279
epoch: 22800, loss: 0.0043, R^2: -9.0517
epoch: 22900, loss: 0.0043, R^2: -9.0803
epoch: 23000, loss: 0.0043, R^2: -9.1014
epoch: 23100, loss: 0.0043, R^2: -9.1311
epoch: 23200, loss: 0.0043, R^2: -9.1586
epoch: 23300, loss: 0.0043, R^2: -9.1736
epoch: 23400, loss: 0.0043, R^2: -9.2096
epoch: 23500, loss: 0.0043, R^2: -9.2331
epoch: 23600, loss: 0.0043, R^2: -9.2581
epoch: 23700, loss: 0.0043, R^2: -9.2766
epoch: 23800, loss: 0.0043, R^2: -9.2903
epoch: 23900, loss: 0.0043, R^2: -9.2758
epoch: 24000, loss: 0.0043, R^2: -9.1949
epoch: 24100, loss: 0.0043, R^2: -8.7469
epoch: 24200, loss: 0.0042, R^2: -7.3713
epoch: 24300, loss: 0.0042, R^2: -6.5890
epoch: 24400, loss: 0.0042, R^2: -7.0720
epoch: 24500, loss: 0.0042, R^2: -7.6964
epoch: 24600, loss: 0.0042, R^2: -8.2028
epoch: 24700, loss: 0.0042, R^2: -8.5974
epoch: 24800, loss: 0.0042, R^2: -8.8538
epoch: 24900, loss: 0.0042, R^2: -9.0612
epoch: 25000, loss: 0.0042, R^2: -9.1873
epoch: 25100, loss: 0.0041, R^2: -9.2905
epoch: 25200, loss: 0.0041, R^2: -9.4218
epoch: 25300, loss: 0.0041, R^2: -9.4044
epoch: 25400, loss: 0.0041, R^2: -9.4644
epoch: 25500, loss: 0.0041, R^2: -9.4459
epoch: 25600, loss: 0.0041, R^2: -9.4576
epoch: 25700, loss: 0.0041, R^2: -9.4099
epoch: 25800, loss: 0.0041, R^2: -9.4411
epoch: 25900, loss: 0.0041, R^2: -9.4235
epoch: 26000, loss: 0.0041, R^2: -9.4040
epoch: 26100, loss: 0.0041, R^2: -9.3837
epoch: 26200, loss: 0.0041, R^2: -9.3491
epoch: 26300, loss: 0.0041, R^2: -9.3139
epoch: 26400, loss: 0.0041, R^2: -9.2739
epoch: 26500, loss: 0.0041, R^2: -9.2438
epoch: 26600, loss: 0.0041, R^2: -9.2039
epoch: 26700, loss: 0.0041, R^2: -9.1616
epoch: 26800, loss: 0.0041, R^2: -9.1250
epoch: 26900, loss: 0.0041, R^2: -9.0562
epoch: 27000, loss: 0.0041, R^2: -9.0385
epoch: 27100, loss: 0.0041, R^2: -8.9928
epoch: 27200, loss: 0.0041, R^2: -8.9336
epoch: 27300, loss: 0.0041, R^2: -8.9435
epoch: 27400, loss: 0.0041, R^2: -8.8865
epoch: 27500, loss: 0.0041, R^2: -8.8282
epoch: 27600, loss: 0.0041, R^2: -8.7933
epoch: 27700, loss: 0.0041, R^2: -8.7472
epoch: 27800, loss: 0.0041, R^2: -8.7464
epoch: 27900, loss: 0.0041, R^2: -8.7031
epoch: 28000, loss: 0.0041, R^2: -8.6399
epoch: 28100, loss: 0.0041, R^2: -8.5974
epoch: 28200, loss: 0.0041, R^2: -8.5528
epoch: 28300, loss: 0.0041, R^2: -8.5265
epoch: 28400, loss: 0.0041, R^2: -8.4908
epoch: 28500, loss: 0.0041, R^2: -8.4488
epoch: 28600, loss: 0.0041, R^2: -8.4245
epoch: 28700, loss: 0.0041, R^2: -8.3903
epoch: 28800, loss: 0.0041, R^2: -8.3547
epoch: 28900, loss: 0.0041, R^2: -8.3265
epoch: 29000, loss: 0.0041, R^2: -8.2999
epoch: 29100, loss: 0.0041, R^2: -8.2628
epoch: 29200, loss: 0.0041, R^2: -8.2418
epoch: 29300, loss: 0.0041, R^2: -8.2016
epoch: 29400, loss: 0.0041, R^2: -8.1720
epoch: 29500, loss: 0.0041, R^2: -8.1272
epoch: 29600, loss: 0.0040, R^2: -8.0824
epoch: 29700, loss: 0.0040, R^2: -8.0462
epoch: 29800, loss: 0.0040, R^2: -8.0369
epoch: 29900, loss: 0.0040, R^2: -8.0168
epoch: 30000, loss: 0.0040, R^2: -7.9990
0.013163033265617595
"""