<!-- # IOSM (IEEE TCSVT 2023) -->
# IOSM
<!-- ### 📖[**Paper**](https://ieeexplore.ieee.org/document/10239514) | 🖼️[**PDF**](./img/lgtd.pdf) | 🎁[**Dataset**](https://zenodo.org/record/6969604) -->

PyTorch codes for "[Inpaint-Outpaint Synergy: Mask Refinement for Trimap-Free Matting](https://ieeexplore.ieee.org/document/10239514)"
<!-- **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**, 2025. -->

- Authors: Xuecheng Li, Yuanjie Zheng*
- Shandong Normal University, Jinan, China

### Abstract
> Image matting is a fundamental task in computer vision that focuses on the precise separation of foreground objects from their backgrounds in images. This process is essential for numerous applications, such as image editing, film production, and augmented reality. Traditional methods often rely on a trimap, a predefined region that helps to distinguish the foreground from the background. However, generating an accurate trimap requires the provision of raw alpha matte, which is labor intensive and prone to drawing errors, limiting the applicability of the relevant methods in practical applications. In this paper, a novel inpaint and outpaint synergy matting approach (IOSM) is proposed to generate masks for image matting tasks without supervision by iterating through the inpaint and outpaint processes, avoiding the dependence on trimap. Specifically, the inpaint process is able to eliminate the false-positive regions present in the initial mask, while the outpaint process reduces the false-negative regions by expanding the pixels to the outer regions. The above process reduces inaccurate regions in the initial mask by means of adversarial updating, providing accurate target information for the subsequent matting stage. By iteratively combining these two processes, a more accurate mask is generated, which is then fed into the mask-to-matte (MTM) module along with the original image to obtain the final alpha matte. This approach allows for the seamless integration of the mask with the original image, improving the matting task and resulting in higher-quality matte outcomes. Experimental results demonstrate that IOSM outperforms other mainstream methods on the AIM-500, Distinctions-646 and PPM-100 datasets.
> 
### Network  
 ![image](/imgs/mpms.png)
 ![image](/imgs/steps.png)
<!-- ## 🧩Install
```
git clone https://github.com/XY-boy/LGTD.git
``` -->
## Environment
 * CUDA 12.4
 * PyTorch 2.0
 * Python 3.9
 
 
## Usage

1. Generate Accurate Binary Mask
    - Open `MPM_Mask_Acquire`
    - Run the following command for iterative optimization from roughprompt to accurate mask.
    ```angular2html
    python scripts/PaintMatting.py --outdir $outdir$ --iters $iter_num$ --steps $diffusion step$ --dataset $dataset$ 
    ```
    - Put the obtained results in the `MPM_MTM_Modules` folder together with the datasets.
2. Use MPM_MTM_Modules
    - Run the `main.py` file for training.
   ```angular2html
    python main.py  --datasets {your datesets location} --fe {frozen encoder?} --norm {is norm your datasets image?}
    ```

<!-- 
## Citation
If you find our work helpful in your research, please consider citing it. Thank you! 😊😊
```
@ARTICLE{xiao2023lgtd,
  author={Xiao, Yi and Yuan, Qiangqiang and Jiang, Kui and Jin, Xianyu and He, Jiang and Zhang, Liangpei and Lin, Chia-wen},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Local-Global Temporal Difference Learning for Satellite Video Super-Resolution}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TCSVT.2023.3312321}
} -->
<!-- ``` -->