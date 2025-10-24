# plume-hunter
Detecting Methane Plumes Onboard Spacecraft

## Towards Methane Detection On Board Satellites [![arXiv](https://img.shields.io/badge/arXiv-2301.12345-b31b1b.svg)](https://arxiv.org/pdf/2509.00626)

<details>
  <summary><b>Abstract</b></summary>
Methane is a potent greenhouse gas and a major driver of climate change, making its timely detection critical for effective mitigation. Machine learning (ML) deployed onboard satellites can enable rapid detection while reducing downlink costs, supporting faster response systems. Conventional methane detection methods often rely on image processing techniques, such as orthorectification to correct geometric distortions and matched filters to enhance plume signals. We introduce a novel approach that bypasses these preprocessing steps by using unorthorectified data (UnorthoDOS). We find that ML models trained on this dataset achieve performance comparable to those trained on orthorectified data. Moreover, we also train models on an orthorectified dataset, showing that they can outperform the matched filter baseline (mag1c). We release model checkpoints and two ML-ready datasets comprising orthorectified and unorthorectified hyperspectral images from the Earth Surface Mineral Dust Source Investigation (EMIT) sensor at https://huggingface.co/datasets/SpaceML/UnorthoDOS , along with code at https://github.com/spaceml-org/plume-hunter.
</details>

### Dataset

The hyperspectral images and corresponding methane plume masks of the UnorthoDOS (Unorthorectified Dataset for On board Satellite methane detection) and its orthorectified counterpart datasets are available for download via <a href="https://huggingface.co/datasets/SpaceML/UnorthoDOS">Hugging Face</a>. Due to storage constraints, the full training and evaluation datasets could not be uploaded. If you're interested in accessing data without methane plumes, please feel free to contact us.

### Code

**Install**

```bash
conda env create --file=environment.yaml
conda activate plume_hunter

pip install -e .
```

## Citation
If you find the code or the UnorthoDOS datasets useful in your research, please consider citing our work.

### 📄 Paper
```
@article{chen2025towards,
  title={Towards Methane Detection Onboard Satellites},
  author={Chen, Maggie and Lambdouar, Hala and Marini, Luca and Mart{\'\i}nez-Ferrer, Laura and Bridges, Chris and Acciarini, Giacomo},
  journal={arXiv preprint arXiv:2509.00626},
  year={2025}
}
```

### 📦 Dataset
```
@misc{UnorthoDOS,
	author       = { Maggie Chen and Hala Lambdouar and Luca Marini and Laura Martínez-Ferrer and Chris Bridges and Giacomo Acciarini },
	title        = { UnorthoDOS (Revision a370bd0) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/SpaceML/UnorthoDOS },
	doi          = { 10.57967/hf/6778 },
	publisher    = { Hugging Face }
}
```
