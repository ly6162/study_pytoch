import  torch
a=torch.rand(3,4)
print("a:",a)
b=torch.rand(4)
print("b:",b)
print("a+b",a+b,"\n",torch.add(a,b))

a=torch.tensor([[1,2],[4,3]])
c=torch.tensor([[1,4],[4,3]])
print("a:",a)
print(a.view(-1,2).data.max(1))
b=torch.tensor([[1,2],[3,4]])
print("b:",b)

print("a*b:",a*b)
print("torch.mm:",torch.mm(a,b))
print("torch.matmul:",torch.matmul(a,b))
print("a@b:",a@b)
c=a*b
print("c.argmax:",c.argmax(dim=13))