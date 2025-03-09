import torch

def test(imports):
    print = imports['print_fn']

    a = torch.arange(6,dtype=torch.float32).reshape(2,3)

    print(torch.mean(a,0,keepdim=True))

    print(torch.mean(a,1,keepdim=True))

    print(torch.mean(a,tuple(range(a.dim())),keepdim=True))

    b = torch.arange(24,dtype=torch.float32).reshape(2,3,4)

    print(torch.mean(b,0,keepdim=True))

    print(torch.mean(b,1,keepdim=True))

    print(torch.mean(b,2,keepdim=True))

    print(torch.mean(b,(0,1),keepdim=True))

    print(torch.mean(b,(1,2),keepdim=True))

    print(torch.mean(b,(0,2),keepdim=True))
