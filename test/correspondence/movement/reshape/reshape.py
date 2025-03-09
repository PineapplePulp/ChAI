import torch

def test(imports):
    print = imports['print_fn']

    a = torch.arange(6,dtype=torch.float32).reshape(2,3)

    print(a.reshape(3,2))
    
    print(a.reshape(6))

    print(a.reshape(1,2,3))

    print(a.reshape(1,1,6))

    print(a.reshape(6,1,1))

    print(a.reshape(2,1,1,3))

    print(a.reshape(3,1,1,2))

    print(a.reshape(1,3,2,1))

