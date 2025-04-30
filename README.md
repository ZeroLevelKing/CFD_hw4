# The 4th homework for CFD

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

计算流体力学第四次课程作业

## 主要内容

- 关于第四次作业的报告
- 关于第四次作业的数值验证源码

## 前置要求

- Python 3.7+
- matplotlib  第三方库
- texlive/simple Tex

## 快速开始

基本使用：

- 在 `/doc` 路径下查看报告的 latex 源码和已经编译完成的.pdf文件
- 在 `/src` 路径下查看 python 源码，可以直接运行 `task1.py` 、`task2.py` 、`task3.py` 来查看数值验证结果。

## 项目结构

```plaintext
project-root/
├── src/                   # 源代码目录
│   ├── common.py          # 包含迭代算法基本函数，网格划分基本函数，可视化基本函数
│   ├── task1.py           # 运行后使用迭代法计算板内稳定的温度场，并绘出等温线
│   ├── task2.py           # 采用不同的松弛因子，比较收敛速度
│   └── task3.py           # 采用不同的网格尺度，观察最佳松弛因子的变化
├── docs/                  # 报告目录
│   ├── picture/           # 图片存放目录
│   └── hw2.tex            # 报告的 latex 源码
|   └── hw2.pdf            # 编译完成的报告
└── ReadMe.md              # 说明文档
```