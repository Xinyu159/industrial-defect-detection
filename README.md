# Industrial Defect Detection — C++/OpenCV

钢材表面缺陷检测(传统机器视觉路线),数据集:**NEU-DET**(东北大学钢材表面缺陷库,6 类 ×300 张官方 / 本仓库配套 1770 张含标注镜像子集,200×200 灰度)。

## Pipeline

1. 预处理:灰度化 → 高斯/中值滤波 → 直方图均衡(低对比度/反光场景)
2. 缺陷分割:阈值(Otsu)/边缘(Canny)+ 形态学连接 / 顶帽变换 + blob —— 按缺陷形态选策略
3. 特征提取 + 分类:连通域分析 + 几何/灰度特征 + 规则分类(后续扩展 SVM)
4. 评估:批量测试 → Precision / Recall / 准确率 + 可视化标注图

> 开发进行中,随阶段推进逐步更新本文档(方法细节、运行命令、实验指标、结果图)。

## 目录结构

```
industrial-defect-detection/
├── data/            # NEU-DET 数据集(不入库,下载见下方指引)
├── include/         # 头文件
├── src/             # 模块源码
├── scripts/         # 批量测试/评估脚本
└── results/         # 标注可视化输出(不入库)
```

## 数据集下载

NEU-DET(1800 张,6 类:裂纹/夹杂/斑块/麻点/氧化铁皮/划痕,含 VOC 标注):
- 官方:http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm
- 本仓库开发使用的镜像子集(1770 张 + VOC XML):[SprAJR/NEU-DET-Steel-Surface-Defect-Detection](https://github.com/SprAJR/NEU-DET-Steel-Surface-Defect-Detection)

## 环境

- Ubuntu 22.04 / g++ 11 / CMake ≥ 3.16 / OpenCV 4.x
