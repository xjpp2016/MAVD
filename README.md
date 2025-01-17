# MAVD
 Aligning First, Then Fusing: A Novel Weakly Supervised Multimodal Violence Detection Method  


 <p align="center">
    <img src=img.png width="800" height="300"/>
</p>

 [Paper](https://arxiv.org/abs/2501.07496)

## Abstract
Weakly supervised violence detection refers to the technique of training models to identify violent segments in videos using only video-level labels. Among these approaches, multimodal violence detection, which integrates modalities such as audio and optical flow, holds great potential. Existing methods in this domain primarily focus on designing multimodal fusion models to address modality discrepancies. In contrast, we take a different approach; leveraging the inherent discrepancies across modalities in violence event representation to propose a novel multimodal semantic feature alignment method. This method sparsely maps the semantic features of local, transient, and less informative modalities ( such as audio and optical flow ) into the more informative RGB semantic feature space. Through an iterative process, the method identifies the suitable no-zero feature matching subspace and aligns the modality-specific event representations based on this subspace, enabling the full exploitation of information from all modalities during the subsequent modality fusion stage. Building on this, we design a new weakly supervised violence detection framework that consists of unimodal multiple-instance learning for extracting unimodal semantic features, multimodal alignment, multimodal fusion, and final detection. Experimental results on benchmark datasets demonstrate the effectiveness of our method, achieving an average precision (AP) of 86.07% on the XD-Violence dataset. Our code is available at https://github.com/xjpp2016/MAVD.

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


## Pre-trained Alignment Models

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


## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@article{jin2025aligning,
  title={Aligning First, Then Fusing: A Novel Weakly Supervised Multimodal Violence Detection Method},
  author={Jin, Wenping and Zhu, Li and Sun, Jing},
  journal={arXiv preprint arXiv:2501.07496},
  year={2025}
}
```

## Acknowledgements
Some of the code is derived from [UR-DMU](https://github.com/henrryzh1/UR-DMU) and [MSBT](https://github.com/shengyangsun/MSBT). We sincerely thank the authors for their valuable contributions.