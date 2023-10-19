# Meta-Learning-Papers-with-Code
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/WangJingyao07/Meta-Learning-Papers-with-Code)

This repository contains a reading list of papers with code on **Meta-Learning** and **Meta-Reinforcement-Learning**, These papers are mainly categorized according to the type of model. In addition, I will separately list papers from important conferences starting from 2023, e.g., NIPS, ICML, ICLR, CVPR etc. **This repository is still being continuously improved. If you have found any relevant papers that need to be included in this repository, please feel free to submit a pull request (PR) or open an issue.**

Each paper may be applicable to one or more types of meta-learning frameworks, including optimization-based and metric-based, and may be applicable to multiple data sources, including image, text, audio, video, and multi-modality. **These are marked in the type column**. In addition, for different tasks and different problems, **we have marked the SOTA algorithm separately**. This is submitted with reference to the leadboard at the time of submission, and will be continuously modified. **We provide a basic introduction to each paper to help you understand the work and core ideas of this article more quickly**.

### Label

🎭 **Different Frameworks**
* ![Meta-Learning](https://img.shields.io/badge/-ML-gray) Meta-Learning.
* ![Meta-Reinforcement-Learning](https://img.shields.io/badge/-MRL-white)  Meta-Reinforcement-Learning.

🎨 **Different Types**
* ![optimization-based](https://img.shields.io/badge/-Optimization-blue) Optimization-based meta-learning approaches acquire a collection of optimal initial parameters, facilitating rapid convergence of a model when adapting to novel tasks.
* ![metric-based](https://img.shields.io/badge/-Metric-red)  Metric-based meta-learning approaches acquire embedding functions that transform instances from various tasks, allowing them to be readily categorized using non-parametric methods.

✨ **Different Data Sources**
* ![Image](https://img.shields.io/badge/-CVimage-brightgreen) Meta-Learning for CV (Images)
* ![Video](https://img.shields.io/badge/-CVvideo-green) Meta-Learning for CV (Videos)
* ![Text](https://img.shields.io/badge/-NLP-pink)  Meta-Learning for NLP
* ![Audio](https://img.shields.io/badge/-Audio-orange)  Meta-Learning for Audio
* ![Multi](https://img.shields.io/badge/-MultiModal-purple)  Meta-Learning for Multi-modal

It is worth noting that the experiments of some frameworks consist of multiple data sources. Our annotations are based on the paper description.

🚩 **I have marked some recommended papers with 🌟/🎈 (SOTA methods/Just my personal preference 😉).**

## Survey.


| Date     | Method                                                       | Type                                                         | Conference  | Paper Title and Paper Interpretation (In Chinese)            | Code                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2019 | [Book of Meta-Learning](https://library.oapen.org/bitstream/handle/20.500.12657/23012/1/1007149.pdf#page=46) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| Book | Meta-Learning (Automated Machine Learning) | None                                                         |
| 2019 | [Learn dynamics](https://arxiv.org/abs/1905.01320) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| arXiv 2019 | Meta-learners' learning dynamics are unlike learners' | None                                                         |
| 2020 | [NLP](https://arxiv.org/abs/2007.09604) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| arXiv 2020 | Meta-learning for few-shot natural language processing: A survey | None                                                         |
| 2020 | [CV-classifier](https://ieeexplore.ieee.org/abstract/document/8951014) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| IEEE Access | A literature survey and empirical study of meta-learning for classifier selection | None                                         |
| 2021 | [Learn 2 Learn](https://library.oapen.org/bitstream/handle/20.500.12657/23012/1/1007149.pdf#page=46) | ![Meta-Learning](https://img.shields.io/badge/-ML-grayk)| arXiv 2021 | Meta-Learning: A Survey | None                                                         |
| 2021 | [Learn 2 Learn 🎈](https://arxiv.org/abs/2004.05439) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| TPAMI | Meta-Learning in Neural Networks: A Survey | None                                                         |
| 2021 | [Learn 2 Learn](https://arxiv.org/abs/2004.05439) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| Artif Intell Rev |  A survey of deep meta-learning | None                                                         |
| 2021 | [Learn 2 Learn](https://www.sciencedirect.com/science/article/abs/pii/S2352154621000024) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| Current Opinion in Behavioral Sciences | Meta-learning in natural and artificial intelligence | None                         |
| 2022 | [Multi-Modal](https://arxiv.org/abs/2004.05439) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| KBS | Multimodality in meta-learning: A comprehensive survey | None                                                         |
| 2022 | [Image Segmentation](https://arxiv.org/abs/2004.05439) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| PR | Meta-seg: A survey of meta-learning for image segmentation | None                                                         |
| 2022 | [Cyberspace Security](https://www.sciencedirect.com/science/article/pii/S2352864822000281) | ![Meta-Learning](https://img.shields.io/badge/-ML-gray)| Digit. Commun. Netw. | Application of meta-learning in cyberspace security: A survey | None                                                         |



