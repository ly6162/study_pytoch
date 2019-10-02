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

#out:batch_size  in:28*28=784
w1, b1 = torch.randn(batch_size, 784, requires_grad=True), torch.zeros(batch_size, requires_grad=True)

w2, b2 = torch.randn(batch_size, batch_size, requires_grad=True), torch.zeros(batch_size, requires_grad=True)
w3, b3 = torch.randn(10, batch_size, requires_grad=True),  torch.zeros(10, requires_grad=True)

#初期化大事
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(x):
    #.t 転置 2d限定 @ =matmul
    #(batch_size,784)
    x = torch.matmul(x,w1.t()) + b1
    #(batch_size,batch_size)
    x = F.relu(x)
    #(batch_size,batch_size)
    x = x@w2.t() + b2
    #(batch_size,batch_size)
    x = F.relu(x)
    x = x@w3.t() + b3
    #(batch_size,10)
    x = F.relu(x)
    # Test set: Average loss: 0.0003, Accuracy: 9351/10000 (93%)
    #x = F.sigmoid(x)
    return x

optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        #data:(batch_size,784)
        logits = forward(data)
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

        logits = forward(data)
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