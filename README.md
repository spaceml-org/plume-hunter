# plume-hunter
Detecting Methane Plumes Onboard Spacecraft

## Methane Detection On Board Satellites from Unorthorectified Imagery [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18996348.svg)](https://doi.org/10.5281/zenodo.24024726) [![arXiv](https://img.shields.io/badge/arXiv-2301.12345-b31b1b.svg)](https://arxiv.org/pdf/2509.00626)

<details>
  <summary><b>Abstract</b></summary>
As a potent greenhouse gas, methane is a major driver of climate change. Its effective mitigation relies on timely detection. Conventional detection methods rely on orthorectification to correct geometric distortions and matched filters to enhance plume signals, which are steps designed for ground processing and poorly suited to onboard execution. We introduce UnorthoDOS, a dataset and approach for training machine learning models directly on unorthorectified hyperspectral imagery, bypassing both orthorectification and matched-filter products. Our UNet models trained on unorthorectified data approach the performance of models trained on orthorectified data (IoU 16.91% vs. 18.47% on all plumes), while both substantially outperform the mag1c matched-filter baseline (IoU 4.76%). We further demonstrate the feasibility of onboard deployment: FP16 compression halves model size with under 0.3% output deviation. The trained ML models and two ML-ready datasets - orthorectified and unorthorectified hyperspectral imagery from the EMIT sensor - are publicly available at https://huggingface.co/datasets/SpaceML/UnorthoDOS, with code at https://github.com/spaceml-org/plume-hunter.
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

```bibtex
@misc{chen_2026_24024726,
  author       = {Chen, Maggie and
                  Lamdouar, Hala and
                  Marini, Luca and
                  Martínez-Ferrer, Laura and
                  Bridges, Chris and
                  Acciarini, Giacomo},
  title        = {Methane Detection On Board Satellites from
                   Unorthorectified Imagery
                  },
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {2026-11-07},
  doi          = {10.5281/zenodo.24024726},
  url          = {https://doi.org/10.5281/zenodo.24024726},
}
```

```bibtex
@inproceedings{chen2025towards,
  title={Towards Methane Detection Onboard Satellites},
  author={Chen, Maggie and Lambdouar, Hala and Marini, Luca and Mart{\'\i}nez-Ferrer, Laura and Bridges, Chris and Acciarini, Giacomo},
  booktitle = {Machine Learning and the Physical Sciences Workshop at NeurIPS 2026, San Diego, California, US},
  url={https://arxiv.org/abs/2509.00626},
  year={2025}
}
```

### 📦 Dataset
```bibtex
@misc{UnorthoDOS,
	author       = { Maggie Chen and Hala Lambdouar and Luca Marini and Laura Martínez-Ferrer and Chris Bridges and Giacomo Acciarini },
	title        = { UnorthoDOS (Revision a370bd0) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/SpaceML/UnorthoDOS },
	doi          = { 10.57967/hf/6778 },
	publisher    = { Hugging Face }
}
```
