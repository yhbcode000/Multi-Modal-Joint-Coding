# A novel multimodal fusion network based on a joint coding model for lane line segmentation.

## Abstract

There has recently been growing interest in utilizing multimodal sensors to achieve robust lane line segmentation. In this paper, we introduce a novel multimodal fusion architecture from an information theory perspective, and demonstrate its practical utility using Light Detection and Ranging (LiDAR) camera fusion networks. In particular, we develop, for the first time, a multimodal fusion network as a joint coding model, where each single node, layer, and pipeline is represented as a channel. The forward propagation is thus equal to the information transmission in the channels. Then, we can qualitatively and quantitatively analyze the effect of different fusion approaches. We argue the optimal fusion architecture is related to the essential capacity and its allocation based on the source and channel. To test this multimodal fusion hypothesis, we progressively determine a series of multimodal models based on the proposed fusion methods and evaluate them on the KITTI and the A2D2 datasets. Our optimal fusion network achieves 85\%+ lane line accuracy and 98.7\%+ overall. The performance gap among the models will inform continuing future research into development of optimal fusion algorithms for the deep multimodal learning community.

## Code base

The whole project is developed based on prject: self-supervised-depth-completion, details can be found in file: [README_base.md](README_base.md).

> Notice: Dataset is different from the code base, with a smaller database, for lane line segmentation task.

## Dataset



## Citation

```

@article{Zou2022,
	title = {A novel multimodal fusion network based on a joint coding model for lane line segmentation},
	volume = {80},
	issn = {15662535},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S1566253521002153},
	doi = {10.1016/j.inffus.2021.10.008},
	journal = {Information Fusion},
	author = {Zou, Zhenhong and Zhang, Xinyu and Liu, Huaping and Li, Zhiwei and Hussain, Amir and Li, Jun},
	month = apr,
	year = {2022},
	note = {arXiv: 2103.11114
Publisher: Elsevier B.V.},
	keywords = {yangsir like, Information theory, Lane line segmentation, Multimodal fusion, Neural Network, Semantic segmentation},
	pages = {167--178},
	file = {Submitted Version:C\:\\Users\\Administrator\\Zotero\\storage\\T24BXLJY\\Zou et al. - 2022 - A novel multimodal fusion network based on a joint.pdf:application/pdf},
}

```