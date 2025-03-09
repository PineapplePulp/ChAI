import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.Softplus()(torch.zeros(2,3))
    print(a)

    b = torch.nn.Softplus()(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Softplus()(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with alpha = -0.001
    a = torch.nn.Softplus(beta=-0.001)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Softplus(beta=-0.001)(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Softplus(beta=-.001)(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with alpha = 10.0
    a = torch.nn.Softplus(beta=10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Softplus(beta=10.0)(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Softplus(beta=10.0)(torch.zeros(10) + 40.0)
    print(c)
    
    a = torch.nn.Softplus(beta=0.0)(torch.zeros(5))
    print(a)