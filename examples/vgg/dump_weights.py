import torch.nn as nn
import torch
import sys
import os

import vgg


from vgg import vgg16

model = vgg16(pretrained=True)

model = model.to(torch.float16)

# Add the scripts directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))

import chai

os.makedirs('models', exist_ok=True)

os.makedirs('models/vgg16', exist_ok=True)
model.chai_dump('models/vgg16','vgg16', with_json=False, verbose=True)

# # print(model.state_dict().keys())
# # print([(n,w.dtype) for (n,w) in model.state_dict().items()])

for vggxx in vgg.model_urls.keys():
    
    print(vggxx)

    if vggxx != 'vgg16':
        print(f'skipping {vggxx}...')
        continue

    model = vgg.__dict__[vggxx](pretrained=True)
    model = model.to(torch.float16)
    model.eval()

    # # Add the scripts directory to the sys.path
    
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))

    # import chai

    # os.makedirs('models/vgg16', exist_ok=True)
    # model.chai_dump('models/vgg16',vggxx, with_json=False, verbose=True)

    # print(f'models/traced_{vggxx}.pt')
    # faux_input = torch.randn(3,720,1280)
    # traced_model = torch.jit.trace(model, faux_input)
    # traced_model.save(f'models/traced_{vggxx}.pt')

    # print(f'models/{vggxx}.pt')
    # sd = model.state_dict()
    # torch.save(sd, f'models/{vggxx}.pt')

    # # print(f'models/traced_{vggxx}.pt')
    # # faux_input = torch.randn(3,720,1280).to(torch.float16)
    # # traced_model = torch.jit.trace(model, faux_input)
    # # torch.save(traced_model,f'models/traced_{vggxx}.pt')

    # print(f'models/script_{vggxx}.pt')
    # script_model = torch.jit.script(model)
    # script_model.save(f'models/script_{vggxx}.pt')

    # test = torch.rand(1,3,720,1280).to(torch.float32)
    # print(model(test).shape)

    model.eval()
    print(f'models/trace_{vggxx}.pt')
    faux_input = torch.rand(1,3,720,1280).to(torch.float16)
    print(model(faux_input).shape)
    trace_model = torch.jit.trace(model,faux_input)
    trace_model.save(f'models/trace_{vggxx}.pt')
    # torch.save(trace_model,f'models/trace_{vggxx}.pt')


    # print(model.state_dict().keys())
    # print([(n,w.dtype) for (n,w) in model.state_dict().items()])
    