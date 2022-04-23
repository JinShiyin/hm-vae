
## Environment 
```
conda create -n hm-vae-env python=3.8
conda activate hm-vae-env
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tqdm
pip install torchgeometry
pip install tensorboard
pip install scipy
pip install pyyaml
pip install opencv-python
pip install matplotlib
```

The above commands may led to torch not work, change to use pip to install pytorch and torchvision
```
conda create -n hm-vae-env python=3.8
conda activate hm-vae-env
conda install cudatoolkit=10.2
conda install osmesa
pip install torch torchvision
pip install tqdm
pip install torchgeometry
pip install tensorboard
pip install scipy
pip install pyyaml
pip install opencv-python
pip install matplotlib==3.2.0
pip install smplx==0.1.26
pip install pyrender==0.1.36
pip uninstall pyopengl
git clone https://github.com/mmatl/pyopengl.git
pip install ./pyopengl
```

You can also follow the following commands to build the environment:
```bash
conda create -n hmvae-env python=3.7
conda activate hmvae-env
conda install cudatoolkit=10.2
conda install pysocks
pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
pip install -r requirements.txt
conda install osmesa
pip uninstall pyopengl
git clone https://github.com/mmatl/pyopengl.git
pip install ./pyopengl
```

## Train hm-vae 
```
python train_motion_vae.py --config ./configs/len64_no_aug_hm_vae.yaml
```

## Motion Completeion
Coming soon. 

## Motion Interpolation 
Coming soon. 