import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms


batch_size=800
learning_rate=0.01
epochs=10

#画像データロード
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)

device=torch.device('cpu')

#初期化はいらない
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, batch_size),
            nn.ReLU(inplace=True),
            nn.Linear(batch_size, batch_size),
            nn.ReLU(inplace=True),
            nn.Linear(batch_size, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        #data:(batch_size,784)
        logits = net.forward(data)
        #target:(batch_size)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        #print(w1.grad.norm(), w2.grad.norm())

        if batch_idx % 100 == 0:
            print('Train Epoch: {} Train rate [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)

        logits = net.forward(data)
        test_loss += criteon(logits, target).item()
        #logits.data:(batch_size,10)
        #max(1):max of rows. max(0):max of column
        pred = logits.data.max(1)[1]
        #.eqは同じなら「true」になる。.sumは「true」の数を統計
        correct += pred.eq(target.data).sum()

    test_loss = test_loss/len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))