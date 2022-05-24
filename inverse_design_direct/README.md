# inverse-NN-clean.py 
[1] 原始模型就是三层全连接网络，经过3万次epoch训练，在训练集上loss收敛到（loss: 0.0028）， 测试集上R2为0.0134。
model = nn.Sequential(
    nn.Linear(6,9),
    nn.Softmax(1),
    nn.Linear(9,12),
    nn.Softmax(1),
    nn.Linear(12,12),
    nn.Softmax(1)
)

[2] 改进模型设计。增加了模型神经元数量，以及BatchNorm1d做数据归一化处理。在1万个epoch训练集loss就比之前3万次loss更低
（loss: 0.0023），但测试集的R2也表现不佳（从之前0.0134变为0.037）。事实证明在这个数据量下模型设计太过于冗余导致过拟合。
model = nn.Sequential(
    nn.Linear(6,9),
    nn.BatchNorm1d(9),
    nn.PReLU(),
    nn.Linear(9,15),
    nn.BatchNorm1d(15),
    nn.PReLU(),
    nn.Linear(15,12),
    nn.BatchNorm1d(12),
    nn.Softmax()
)

[3] 根据以上表现，继续改进模型网络，将中间隐藏层数量加到16个，输出层再缩小到12个。整个模型增大后，训练2万个epoch就达到
之前3万epoch的训练集loss（loss: 0.0028），测试集的R2也表现更好（从之前0.0134变为0.0126）。分析原因可能是增加了模型参数量
后与训练数据集达到更好的匹配，表达能力更强。此外，模型设计上有一个放大再缩小的过程，增加了特征选择范围，模型的鲁棒性更好。

model = nn.Sequential(
    nn.Linear(6,9),
    nn.Softmax(1),
    nn.Linear(9,16),
    nn.Softmax(1),
    nn.Linear(16,12),
    nn.Softmax(1)
)


# inverse-NN.py 
类似的，将中间隐藏层数量加到16个，输出层再缩小到12个。整个模型增大后，训练3万个epoch训练集loss（loss: 0.0040，低于之前0.0042），
测试集的R2也表现更好（从之前0.0153变为0.0136）。
model = nn.Sequential(
    nn.Linear(6,9),
    nn.Softmax(1),
    nn.Linear(9,16),
    nn.Softmax(1),
    nn.Linear(16,12),
    nn.Softmax(1)
)