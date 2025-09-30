<br>
<p align="center">
<h1 align="center"><strong> FreqPolicy: Frequency Autoregressive Visuomotor Policy with Continuous Tokens
</strong></h1>
  <p align="center">
      <strong><span style="color: red;">NeurIPS 2025</span></strong>
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
  <br><br>
  <sup>1</sup>ShanghaiTech University&emsp;
  <sup>2</sup>The University of Hong Kong<br>
  <sup>3</sup>Nanyang Technological University&emsp;
  <sup>4</sup>The Chinese University of Hong Kong
  </p>
</p>

  

<p align="center">
  <a href="https://freq-policy.github.io/"><b>📖 Project Page</b></a> |
  <a href="https://arxiv.org/pdf/2506.01583"><b>📄 Paper Link</b></a> |
</p>


## 📣 News
- [2/27/2025] 🎉🎉🎉FreqPolicy has been accepted by NeurIPS 2025!!!🎉🎉🎉

## 😲 Results
Please refer to our [homepage](https://freq-policy.github.io/) for more thrilling results!


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