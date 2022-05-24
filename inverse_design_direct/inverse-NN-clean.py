
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings('ignore')
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


toy_alloy_data = pd.read_csv("../Mercury/data/toydata_clean.csv")

X_col = [f'element {i}' for i in range(1,13)]
Y_col = [f'property {i}' for i in range(1,7)]

X = toy_alloy_data[X_col].values
Y = toy_alloy_data[Y_col].values

# scale Y
scaler = StandardScaler()
Y = scaler.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# move tensors to GPU (or CPU)
X_train = torch.tensor(X_train).to(device)
Y_train = torch.tensor(Y_train).to(device)
X_test = torch.tensor(X_test).to(device)
Y_test = torch.tensor(Y_test).to(device)


# In[3]:


# model = nn.Sequential(
#     nn.Linear(6,9),
#     nn.BatchNorm1d(9),
#     nn.PReLU(),
#     nn.Linear(9,15),
#     nn.BatchNorm1d(15),
#     nn.PReLU(),
#     nn.Linear(15,12),
#     nn.BatchNorm1d(12),
#     nn.Softmax()
# )

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
            with torch.no_grad():
                print(mean_absolute_error(model(Y_test).cpu().numpy(), X_test.cpu().numpy()))
            if len(loss_hist) > 2 and torch.abs((loss_hist[-1]-loss_hist[-2])/loss_hist[-1]) < 1e-4:
                break
    return m, loss_hist, R2_hist


# In[5]:


m, loss_hist, R2_hist= train(model, Y_train, X_train, max_iter=20000)


# In[6]:


plt.plot(loss_hist)


# In[ ]:


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
epoch: 100, loss: 0.0504, R^2: -3936.4779
0.13292016356712397
epoch: 200, loss: 0.0455, R^2: -3298.3205
0.12649524729274145
epoch: 300, loss: 0.0395, R^2: -2641.3490
0.1172917483189251
epoch: 400, loss: 0.0337, R^2: -1929.6789
0.10630967422351055
epoch: 500, loss: 0.0285, R^2: -1334.8157
0.09598209342379972
epoch: 600, loss: 0.0239, R^2: -897.7804
0.08659509833798042
epoch: 700, loss: 0.0204, R^2: -601.2009
0.07809599020985697
epoch: 800, loss: 0.0181, R^2: -412.5129
0.07124087335362468
epoch: 900, loss: 0.0166, R^2: -299.1337
0.06616974234975499
epoch: 1000, loss: 0.0156, R^2: -230.8057
0.06232043468928832
epoch: 1100, loss: 0.0150, R^2: -187.0035
0.05944409488753749
epoch: 1200, loss: 0.0146, R^2: -156.8156
0.05725162393215146
epoch: 1300, loss: 0.0142, R^2: -134.4435
0.05554424497993438
epoch: 1400, loss: 0.0139, R^2: -117.5158
0.054159733117751185
epoch: 1500, loss: 0.0136, R^2: -103.7561
0.05297096808530313
epoch: 1600, loss: 0.0133, R^2: -92.1219
0.051870236151427156
epoch: 1700, loss: 0.0131, R^2: -82.4748
0.050727052773285354
epoch: 1800, loss: 0.0128, R^2: -75.0116
0.04945647665108407
epoch: 1900, loss: 0.0124, R^2: -69.2799
0.048019950545213096
epoch: 2000, loss: 0.0120, R^2: -63.2645
0.04637556126308454
epoch: 2100, loss: 0.0116, R^2: -55.3999
0.04454277899038437
epoch: 2200, loss: 0.0112, R^2: -47.2856
0.04278002919291125
epoch: 2300, loss: 0.0110, R^2: -40.3914
0.041311303634116184
epoch: 2400, loss: 0.0107, R^2: -35.0884
0.040142722208772554
epoch: 2500, loss: 0.0105, R^2: -31.1758
0.03917916746124469
epoch: 2600, loss: 0.0103, R^2: -28.2916
0.038354239572395865
epoch: 2700, loss: 0.0100, R^2: -26.1657
0.03762653116871654
epoch: 2800, loss: 0.0097, R^2: -24.5206
0.036843491623118056
epoch: 2900, loss: 0.0089, R^2: -24.7080
0.03568637588106993
epoch: 3000, loss: 0.0075, R^2: -24.2893
0.03298554172553291
epoch: 3100, loss: 0.0069, R^2: -21.7551
0.030861155683267128
epoch: 3200, loss: 0.0064, R^2: -18.8646
0.029207424300754178
epoch: 3300, loss: 0.0061, R^2: -16.1964
0.0279470825234855
epoch: 3400, loss: 0.0059, R^2: -14.0165
0.026882760958246293
epoch: 3500, loss: 0.0057, R^2: -12.3766
0.02594896272845369
epoch: 3600, loss: 0.0055, R^2: -11.1773
0.02513728065532012
epoch: 3700, loss: 0.0054, R^2: -10.2865
0.024476376316939847
epoch: 3800, loss: 0.0053, R^2: -9.5846
0.0239298646574513
epoch: 3900, loss: 0.0052, R^2: -8.9806
0.023455350882909804
epoch: 4000, loss: 0.0051, R^2: -8.4171
0.023003671284221174
epoch: 4100, loss: 0.0051, R^2: -7.8677
0.022582392819841646
epoch: 4200, loss: 0.0050, R^2: -7.3287
0.022185419429701463
epoch: 4300, loss: 0.0049, R^2: -6.8048
0.021830383838914256
epoch: 4400, loss: 0.0049, R^2: -6.3030
0.021511639636723128
epoch: 4500, loss: 0.0049, R^2: -5.8334
0.021226037877006554
epoch: 4600, loss: 0.0048, R^2: -5.4046
0.02097188391848452
epoch: 4700, loss: 0.0048, R^2: -5.0176
0.020722437813191778
epoch: 4800, loss: 0.0047, R^2: -4.6842
0.020493079647081122
epoch: 4900, loss: 0.0047, R^2: -4.4545
0.02027374205224617
epoch: 5000, loss: 0.0046, R^2: -4.2070
0.019914244721326486
epoch: 5100, loss: 0.0045, R^2: -3.8541
0.019599936342631828
epoch: 5200, loss: 0.0044, R^2: -3.5060
0.019241449839558635
epoch: 5300, loss: 0.0044, R^2: -3.2101
0.018935163961466907
epoch: 5400, loss: 0.0044, R^2: -2.9721
0.01870932758867418
epoch: 5500, loss: 0.0043, R^2: -2.7781
0.018532213284326953
epoch: 5600, loss: 0.0043, R^2: -2.6143
0.018386140936271746
epoch: 5700, loss: 0.0043, R^2: -2.4708
0.018264403953452007
epoch: 5800, loss: 0.0043, R^2: -2.3411
0.01815975938650545
epoch: 5900, loss: 0.0043, R^2: -2.2208
0.01807596014457793
epoch: 6000, loss: 0.0043, R^2: -2.1070
0.0180004633300646
epoch: 6100, loss: 0.0042, R^2: -1.9981
0.01793009614701508
epoch: 6200, loss: 0.0042, R^2: -1.8936
0.01786371724153704
epoch: 6300, loss: 0.0042, R^2: -1.7933
0.017802923624859926
epoch: 6400, loss: 0.0042, R^2: -1.6969
0.017745038042541104
epoch: 6500, loss: 0.0042, R^2: -1.6047
0.017687330383795538
epoch: 6600, loss: 0.0042, R^2: -1.5183
0.017631185604136696
epoch: 6700, loss: 0.0042, R^2: -1.4462
0.01758103107827254
epoch: 6800, loss: 0.0041, R^2: -1.4105
0.017565337200927875
epoch: 6900, loss: 0.0041, R^2: -1.2957
0.01747849256486152
epoch: 7000, loss: 0.0041, R^2: -1.1916
0.017382977738899328
epoch: 7100, loss: 0.0041, R^2: -1.0997
0.01729230420856542
epoch: 7200, loss: 0.0040, R^2: -1.0176
0.01720188717709328
epoch: 7300, loss: 0.0040, R^2: -0.9448
0.01712019170185791
epoch: 7400, loss: 0.0040, R^2: -0.8799
0.017048755041375038
epoch: 7500, loss: 0.0040, R^2: -0.8216
0.016976182878756175
epoch: 7600, loss: 0.0040, R^2: -0.7698
0.016895447805456668
epoch: 7700, loss: 0.0040, R^2: -0.7213
0.016853877390755875
epoch: 7800, loss: 0.0040, R^2: -0.6773
0.01680355305988163
epoch: 7900, loss: 0.0039, R^2: -0.6356
0.016749894120800354
epoch: 8000, loss: 0.0039, R^2: -0.5957
0.01669989872696225
epoch: 8100, loss: 0.0039, R^2: -0.5556
0.016648363589918527
epoch: 8200, loss: 0.0039, R^2: -0.5132
0.016602441965150618
epoch: 8300, loss: 0.0039, R^2: -0.4700
0.016563268283286765
epoch: 8400, loss: 0.0039, R^2: -0.4270
0.016527793163791507
epoch: 8500, loss: 0.0038, R^2: -0.3846
0.01649003006243566
epoch: 8600, loss: 0.0038, R^2: -0.3368
0.016425242468444675
epoch: 8700, loss: 0.0038, R^2: -0.2825
0.016299261201173604
epoch: 8800, loss: 0.0037, R^2: -0.2288
0.016132484135954212
epoch: 8900, loss: 0.0037, R^2: -0.1791
0.015960485127255807
epoch: 9000, loss: 0.0037, R^2: -0.1340
0.01576405114636121
epoch: 9100, loss: 0.0036, R^2: -0.0909
0.015590265403671583
epoch: 9200, loss: 0.0036, R^2: -0.0564
0.01540808481107338
epoch: 9300, loss: 0.0035, R^2: -0.0266
0.015356038031256722
epoch: 9400, loss: 0.0035, R^2: 0.0031
0.015281551342577237
epoch: 9500, loss: 0.0035, R^2: 0.0333
0.015238590990440719
epoch: 9600, loss: 0.0035, R^2: 0.0622
0.015183359584883893
epoch: 9700, loss: 0.0035, R^2: 0.0899
0.015107214260188512
epoch: 9800, loss: 0.0035, R^2: 0.1162
0.015050599644037904
epoch: 9900, loss: 0.0034, R^2: 0.1412
0.014980692634376897
epoch: 10000, loss: 0.0034, R^2: 0.1646
0.01492548029454746
epoch: 10100, loss: 0.0034, R^2: 0.1869
0.014864466424300954
epoch: 10200, loss: 0.0034, R^2: 0.2076
0.014815861880510506
epoch: 10300, loss: 0.0034, R^2: 0.2271
0.0147563613074079
epoch: 10400, loss: 0.0034, R^2: 0.2449
0.014713146866902836
epoch: 10500, loss: 0.0034, R^2: 0.2615
0.014672207751498743
epoch: 10600, loss: 0.0034, R^2: 0.2765
0.014635313150057375
epoch: 10700, loss: 0.0033, R^2: 0.2897
0.014593687083925612
epoch: 10800, loss: 0.0033, R^2: 0.3014
0.014554523851388737
epoch: 10900, loss: 0.0033, R^2: 0.3118
0.014517304045318332
epoch: 11000, loss: 0.0033, R^2: 0.3210
0.01450451599521905
epoch: 11100, loss: 0.0033, R^2: 0.3292
0.01444628824140366
epoch: 11200, loss: 0.0033, R^2: 0.3365
0.014413857338754615
epoch: 11300, loss: 0.0033, R^2: 0.3431
0.014381049498707594
epoch: 11400, loss: 0.0033, R^2: 0.3490
0.014345737251141409
epoch: 11500, loss: 0.0033, R^2: 0.3544
0.014315835863304482
epoch: 11600, loss: 0.0033, R^2: 0.3593
0.014283305018799939
epoch: 11700, loss: 0.0033, R^2: 0.3638
0.014250723937897591
epoch: 11800, loss: 0.0033, R^2: 0.3679
0.014234192902522683
epoch: 11900, loss: 0.0033, R^2: 0.3717
0.014173495794287887
epoch: 12000, loss: 0.0032, R^2: 0.3752
0.014135042971521732
epoch: 12100, loss: 0.0032, R^2: 0.3784
0.014093963790688469
epoch: 12200, loss: 0.0032, R^2: 0.3814
0.014056411990378522
epoch: 12300, loss: 0.0032, R^2: 0.3842
0.0140222227362237
epoch: 12400, loss: 0.0032, R^2: 0.3867
0.013993054009163295
epoch: 12500, loss: 0.0032, R^2: 0.3892
0.013961002416973532
epoch: 12600, loss: 0.0032, R^2: 0.3914
0.01394032222370395
epoch: 12700, loss: 0.0032, R^2: 0.3935
0.013920328315408294
epoch: 12800, loss: 0.0032, R^2: 0.3955
0.013906632796936063
epoch: 12900, loss: 0.0032, R^2: 0.3975
0.013890687465759536
epoch: 13000, loss: 0.0032, R^2: 0.3996
0.013877263567164086
epoch: 13100, loss: 0.0032, R^2: 0.4017
0.013866733126612336
epoch: 13200, loss: 0.0031, R^2: 0.4039
0.01385569109155145
epoch: 13300, loss: 0.0031, R^2: 0.4060
0.013846866241864556
epoch: 13400, loss: 0.0031, R^2: 0.4081
0.013834573856632925
epoch: 13500, loss: 0.0031, R^2: 0.4102
0.013852329469996883
epoch: 13600, loss: 0.0031, R^2: 0.4122
0.013804781945639518
epoch: 13700, loss: 0.0031, R^2: 0.4143
0.013786787838616602
epoch: 13800, loss: 0.0031, R^2: 0.4163
0.013769941611592984
epoch: 13900, loss: 0.0031, R^2: 0.4182
0.013758652229987628
epoch: 14000, loss: 0.0031, R^2: 0.4198
0.01375588811107359
epoch: 14100, loss: 0.0031, R^2: 0.4206
0.01372937983611297
epoch: 14200, loss: 0.0031, R^2: 0.4230
0.013696410587655585
epoch: 14300, loss: 0.0030, R^2: 0.4252
0.013671273448320823
epoch: 14400, loss: 0.0030, R^2: 0.4275
0.013645224959781469
epoch: 14500, loss: 0.0030, R^2: 0.4297
0.013618026124090318
epoch: 14600, loss: 0.0030, R^2: 0.4318
0.013591161992036553
epoch: 14700, loss: 0.0030, R^2: 0.4338
0.01356616118964102
epoch: 14800, loss: 0.0030, R^2: 0.4354
0.013518823861371751
epoch: 14900, loss: 0.0030, R^2: 0.4371
0.01351610598287778
epoch: 15000, loss: 0.0030, R^2: 0.4385
0.013491045293102848
epoch: 15100, loss: 0.0030, R^2: 0.4395
0.013466397572451418
epoch: 15200, loss: 0.0030, R^2: 0.4403
0.01343487251638141
epoch: 15300, loss: 0.0030, R^2: 0.4407
0.013419021949304737
epoch: 15400, loss: 0.0030, R^2: 0.4408
0.013396075054844964
epoch: 15500, loss: 0.0030, R^2: 0.4404
0.013372694387838231
epoch: 15600, loss: 0.0030, R^2: 0.4399
0.013363446417798559
epoch: 15700, loss: 0.0030, R^2: 0.4382
0.013326854122763873
epoch: 15800, loss: 0.0030, R^2: 0.4364
0.013303725524521115
epoch: 15900, loss: 0.0030, R^2: 0.4343
0.013279565780698969
epoch: 16000, loss: 0.0029, R^2: 0.4323
0.013255038325299095
epoch: 16100, loss: 0.0029, R^2: 0.4299
0.013231344074326315
epoch: 16200, loss: 0.0029, R^2: 0.4278
0.013209142416212622
epoch: 16300, loss: 0.0029, R^2: 0.4257
0.013196205803806043
epoch: 16400, loss: 0.0029, R^2: 0.4238
0.01318748680534682
epoch: 16500, loss: 0.0029, R^2: 0.4222
0.013199265507868957
epoch: 16600, loss: 0.0029, R^2: 0.4198
0.01318139629918609
epoch: 16700, loss: 0.0029, R^2: 0.4177
0.0131850878679625
epoch: 16800, loss: 0.0029, R^2: 0.4154
0.013190955291821798
epoch: 16900, loss: 0.0029, R^2: 0.4129
0.013190921760332363
epoch: 17000, loss: 0.0029, R^2: 0.4096
0.013188727585749944
epoch: 17100, loss: 0.0029, R^2: 0.4054
0.013179130690549039
epoch: 17200, loss: 0.0029, R^2: 0.4000
0.013164803615090771
epoch: 17300, loss: 0.0029, R^2: 0.3932
0.013138426426308832
epoch: 17400, loss: 0.0029, R^2: 0.3855
0.01311417438360798
epoch: 17500, loss: 0.0028, R^2: 0.3794
0.013152737697961596
epoch: 17600, loss: 0.0028, R^2: 0.3676
0.013073653771333953
epoch: 17700, loss: 0.0028, R^2: 0.3570
0.013062135493792737
epoch: 17800, loss: 0.0028, R^2: 0.3454
0.013044431681667546
epoch: 17900, loss: 0.0028, R^2: 0.3315
0.01302416955547126
epoch: 18000, loss: 0.0028, R^2: 0.3153
0.013006049667769965
epoch: 18100, loss: 0.0028, R^2: 0.2965
0.012963229720275917
epoch: 18200, loss: 0.0028, R^2: 0.2756
0.012967452459578798
epoch: 18300, loss: 0.0028, R^2: 0.2523
0.012948900195057142
epoch: 18400, loss: 0.0028, R^2: 0.2286
0.012918171956944368
epoch: 18500, loss: 0.0028, R^2: 0.2060
0.012915828467048529
epoch: 18600, loss: 0.0028, R^2: 0.1848
0.012900889610069508
epoch: 18700, loss: 0.0028, R^2: 0.1659
0.012903716686730496
epoch: 18800, loss: 0.0028, R^2: 0.1485
0.012878472741102133
epoch: 18900, loss: 0.0028, R^2: 0.1322
0.01286932103630374
epoch: 19000, loss: 0.0028, R^2: 0.1166
0.012854996194403613
epoch: 19100, loss: 0.0028, R^2: 0.1025
0.012853216981849486
epoch: 19200, loss: 0.0028, R^2: 0.0877
0.012847322492260982
epoch: 19300, loss: 0.0028, R^2: 0.0732
0.012841013030850508
epoch: 19400, loss: 0.0028, R^2: 0.0590
0.012844341554528472
epoch: 19500, loss: 0.0028, R^2: 0.0423
0.01283023638990987
epoch: 19600, loss: 0.0028, R^2: 0.0245
0.012825443562760315
epoch: 19700, loss: 0.0028, R^2: 0.0045
0.012821315770736736
epoch: 19800, loss: 0.0028, R^2: -0.0202
0.012815588504482896
epoch: 19900, loss: 0.0028, R^2: -0.0518
0.012808287715670918
epoch: 20000, loss: 0.0028, R^2: -0.0864
0.012665170285549018
"""