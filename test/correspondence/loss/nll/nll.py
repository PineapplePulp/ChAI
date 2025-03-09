import torch

def test(imports):
    print = imports['print_fn']

    loss_fn = torch.nn.NLLLoss()

    inputs = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]])
    targets = torch.tensor([0, 1])  # Class indices
    loss = loss_fn(inputs, targets)
    print(loss)

    # inputs = torch.tensor([[-0.1, -0.2, -2.0], [-0.3, -0.8, -1.5]])
    # targets = torch.tensor([2, 0])
    # loss = loss_fn(inputs, targets)
    # print(loss)

    # inputs = torch.tensor([
    #     [-0.2, -0.4, -0.6, -0.8],
    #     [-1.0, -0.5, -0.2, -0.1],
    #     [-3.0, -2.0, -1.0, -0.5]
    # ])
    # targets = torch.tensor([1, 2, 3])
    # loss = loss_fn(inputs, targets)
    # print(loss)
