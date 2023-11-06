# Order-preserving Consistency Regularization for Domain Adaptation and Generalization

## How to install

This code is based on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation to install `dassl`.

## How to run

Please follow the steps below before running the script

- modify `DATA` and `DASSL` in `*.sh` based on the paths on your computer
- activate the `dassl` environment via `conda activate dassl`
- `cd` to `scripts/`


### Domain Generalization
```bash
# PACS | Demix Loss
bash dg_demix.sh
```

## Bibtex
```@inproceedings{jing2023order,
title={Order-preserving Consistency Regularization for Domain Adaptation and Generalization},
author={Jing, Mengmeng and Zhen, Xiantong and Li, Jingjing and Snoek, Cees GM},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={18916--18927},
year={2023}
}
```
