<br>
<p align="center">
<h1 align="center"><strong> FreqPolicy: Frequency Autoregressive Visuomotor Policy with Continuous Tokens
</strong></h1>
  <p align="center">
      <strong><span style="color: red;">NIPS 2025</span></strong>
    <br>
   <a href='https://ymzhong66.github.io' target='_blank'>Yiming Zhong</a>&emsp;
   <a href='https://lym29.github.io/' target='_blank'>Yumeng Liu</a>&emsp;
   <a href='https://xiaochy.github.io/' target='_blank'>Chuyang Xiao</a>&emsp;
   <a href='https://yizhifengyeyzm.github.io/' target='_blank'>Zemin Yang</a>&emsp;
   <a href='https://wang-youzhuo.github.io/' target='_blank'>Youzhuo Wang</a>&emsp;
   <a href='https://github.com/csyufei' target='_blank'>Yufei Zhu</a>&emsp;
   <a href='https://shiye21.github.io/' target='_blank'>Ye Shi</a>&emsp;
   <a href='https://yujingsun.github.io/' target='_blank'>Yujing Sun</a>&emsp;
   <a href='https://xingezhu.me/aboutme.html' target='_blank'>Xinge Zhu</a>&emsp;
   <a href='https://yuexinma.me' target='_blank'>Yuexin Ma</a>&emsp;
    <br>
    ShanghaiTech University
    The University of Hong Kong
    Nanyang Technological University
    The Chinese University of Hong Kong    
    <br>
    <br>
  </p>
</p>

  

<p align="center">
  <a href="https://dexgraspanything.github.io/"><b>📖 Project Page</b></a> |
  <a href="https://arxiv.org/pdf/2503.08257"><b>📄 Paper Link</b></a> |
</p>

</div>

>  We present DexGrasp Anything, consistently surpassing previous dexterous grasping generation methods across five benchmarks. Visualization of our method's results are shown on the left.

<div align="center">
    <img src="image1.png" alt="Directional Weight Score" class="blend-img-background center-image" style="max-width: 100%; height: auto;" />
</div>

## 📣 News
- [2/27/2025] 🎉🎉🎉DexGraspAnything has been accepted by CVPR 2025!!!🎉🎉🎉

## 😲 Results
Please refer to our [homepage](https://dexgraspanything.github.io/) for more thrilling results!

# 📚 Datasets

In our data processing, the rotation and translation formula is \( Y = a(x + b) \), where \( Y \) represents the shadow hand after rotation and translation, and \( x \) is the original hand. It is important to emphasize that we use \( Y = a(x + b) \) rather than \( Y = ax + b \). This formulation allows us to conveniently transfer the rotation to the object, i.e., \(O = a^T * O \), enabling more flexible manipulation.

| **Datasets** | **Huggingface link** | **Google drive Link (Format Compatible with Our Dataloader)** | **Paper** |
|:---:|:---:|:---:|:---:|
| **DGA** | [Huggingface](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/Dexgraspanyting.tar.gz) | [Datasets](https://drive.google.com/file/d/1of-m2qxVqgWf-nHV6oVHPk1z4NWVrEPV/view?usp=drive_link) | ✅[DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](https://arxiv.org/pdf/2503.08257) |
| *Realdex*           | [Huggingface](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/Realdex.zip) | [Datasets](https://drive.google.com/file/d/12rgyyKg07PmY6jzl7pMocA4o5ikLFuOA/view)                     | ✅[RealDex: Towards Human-like Grasping for Robotic Dexterous Hand](https://arxiv.org/abs/2402.13853)                                                    |
| *DexGraspNet*          | [Huggingface](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/DexGraspNet.zip) | [Datasets](https://drive.google.com/file/d/1FHJxEDl2jegOpq-g4KZ4eEVvM3gqDQCh/view)                     | ✅[DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation](https://arxiv.org/abs/2210.02697)                                                    |
| *UniDexGrasp*           | [Huggingface](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/UniDexGrasp.zip) | [Datasets](https://drive.google.com/file/d/1-nPUP14x0VOfIqQwYU-hc-WhUaPBxEQ7/view)                     | ✅[UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy](https://arxiv.org/abs/2303.00938)                                                    |
| *MultiDex*             | [Huggingface](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/MultiDex_UR.zip) | [Datasets](https://drive.google.com/file/d/1wHdWLfvxWjpFBV_Ld-j4DwNXAr1UMERf/view)                     | ✅[GenDexGrasp: Generalizable Dexterous Grasping](https://arxiv.org/abs/2210.00722)                                                    |
| *DexGRAB(Retargeting from Grab)*            | [Huggingface](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/DexGRAB.zip) | [Datasets](https://drive.google.com/file/d/1Xmgw-c3lrkab2NIs_1i0Hq95I0Y4Sp8n/view)                     | ✅[Grab: A dataset of whole-body human grasping of objects.](https://arxiv.org/abs/2008.11200)                                                    |

# 🦾 Make your own dataset
- First, your dataset should contain mesh files of objects. You can create an `object_pcds_nors.pkl` file by sampling point clouds from these meshes. Modify the paths in `Process_your_dataset/make_obj_pcds.py` to build your dataset.
    ```bash
    python Process_your_dataset/make_obj_pcds.py
    ```
- Based on your data pose format (such as translation, rotation, qpose, scale), `create a corresponding .pt file`. You can refer to the input examples in the datasets folder to build it. You need to understand the pose format of your dataset. Pay special attention to whether the translation, rotation, and scale are applied to the object point cloud or to the robotic hand. You can also modify your dataloader to ensure that the input poses and object point cloud data visualize correctly.

- During the testing phase, you need to generate a URDF file for all meshes. You can use `Process_your_dataset/make_obj_urdf.py` to generate them (recommended to place in the same directory level as the mesh files).
    ```bash
    python Process_your_dataset/make_obj_urdf.py
    ```

## 🛠️ Setup
- 1. Create a new `conda` environemnt and activate it.（My CUDA version (nvcc --version) is 11.7）

    ```bash
    conda create -n DGA python=3.8
    conda activate DGA
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

- 2. Install the required packages.
    You can change TORCH_CUDA_ARCH_LIST according to your GPU architecture.
    ```bash
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" pip install -r requirements.txt
    ```
    Please install in an environment with a GPU, otherwise it will error.
    ```bash
    cd src
    git clone https://github.com/wrc042/CSDF.git
    cd CSDF
    pip install -e .
    cd ..
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    git checkout tags/v0.7.2  
    FORCE_CUDA=1  TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"  python setup.py install
    cd ..
    ```
- 3. Install the Isaac Gym
    Follow the [official installation guide](https://developer.nvidia.com/isaac-gym) to install Isaac Gym and its dependencies.
    You will get a folder named `IsaacGym_Preview_4_Package.tar.gz` put it in ./src/IsaacGym_Preview_4_Package.tar.gz
    ```bash
    tar -xzvf IsaacGym_Preview_4_Package.tar.gz
    cd isaacgym/python
    pip install -e .
    ```

Before training and testing, please ensure that you set the dataset path, model size, whether to use LLM, sampling method, and other parameters in `configs`.

### Train

- Train with a single GPU

    ```bash
    bash scripts/grasp_gen_ur/train.sh ${EXP_NAME}
    ```

- Train with multiple GPUs

    ```bash
    bash scripts/grasp_gen_ur/train_ddm.sh ${EXP_NAME}
    ```

### Sample

```bash
bash scripts/grasp_gen_ur/sample.sh ${exp_dir} [OPT]
# e.g., Running without Physics-Guided Sampling:   bash scripts/grasp_gen_ur/sample.sh /outputs/exp_dir [OPT]
# e.g., Running with Physics-Guided Sampling:   bash scripts/grasp_gen_ur/sample.sh /outputs/exp_dir OPT
```
- `[OPT]` is an optional parameter for Physics-Guided Sampling.

### Test 

First, you need to run `scripts/grasp_gen_ur/sample.sh` to sample some results. 
You also need to set the dataset file paths in `/envs/tasks/grasp_test_force_shadowhand.py` and /scripts/grasp_gen_ur/test.py`. 
Then, we will compute quantitative metrics using these sampled results.

```bash
bash scripts/grasp_gen_ur/test.sh ${EVAL_DIR} 
# e.g., bash scripts/grasp_gen_ur/test.sh  /outputs/exp_dir/eval/final/2025-03-16_19-15-31
```
<!-- --- -->



## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 

## 💓 Acknowledgement

We would like to acknowledge that some codes are borrowed from [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy), [DP](https://github.com/real-stanford/diffusion_policy), [MAR](https://github.com/LTH14/mar), [FAR](https://github.com/yuhuUSTC/FAR). We appreciate the authors for their great contributions to the community and for open-sourcing their code.

## 🖊️ Citation
```
@article{zhong2025freqpolicy,
  title={FreqPolicy: Frequency Autoregressive Visuomotor Policy with Continuous Tokens},
  author={Zhong, Yiming and Liu, Yumeng and Xiao, Chuyang and Yang, Zemin and Wang, Youzhuo and Zhu, Yufei and Shi, Ye and Sun, Yujing and Zhu, Xinge and Ma, Yuexin},
  journal={arXiv preprint arXiv:2506.01583},
  year={2025}
}
```