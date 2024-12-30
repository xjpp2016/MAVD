# MAVD
 Aligning First, Then Fusing: A Novel Weakly-Supervised Multimodal Violence Detection Method


 <p align="center">
    <img src=img.png width="800" height="300"/>
</p>


## Requirements  

    python
    torch
    numpy
    tqdm
    scikit-learn
    einops


## Training

    python get_models.py 

## Testing

    python get_result.py

## Directly Utilizing the Pre-Trained Multimodal Alignment Model (For Enhancing the Performance of Multimodal Fusion Models)

    import torch

    # Load the pre-trained models
    v_net = torch.jit.load("./saved_models/pt/v_model.pt").cuda()
    a_net = torch.jit.load( "./saved_models/pt/a_model.pt").cuda()
    f_net = torch.jit.load("./saved_models/pt/f_model.pt").cuda()

    va_net = torch.jit.load("./saved_models/pt/va_model.pt").cuda()
    vf_net = torch.jit.load("./saved_models/pt/vf_model.pt").cuda()

    # Get the alligned features
    feature_RGB = v_net(feature_RGB from backbone)
    feature_audio = va_net(a_net(feature_audio from backbone))
    feature_flow = vf_net(f_net(feature_flow from backbone))
