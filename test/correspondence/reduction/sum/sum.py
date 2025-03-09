import torch

def test(imports):
    print = imports['print_fn']

    a = torch.arange(6,dtype=torch.float32).reshape(2,3)

    print(a.sum(0,keepdim=True))

    print(a.sum(1,keepdim=True))

    print(torch.sum(a,tuple(range(a.dim())),keepdim=True))

    b = torch.arange(24,dtype=torch.float32).reshape(2,3,4)

    print(b.sum(0,keepdim=True))

    print(b.sum(1,keepdim=True))

    print(b.sum(2,keepdim=True))

    print(b.sum((0,1),keepdim=True))

    print(b.sum((1,2),keepdim=True))

    print(b.sum((0,2),keepdim=True))

