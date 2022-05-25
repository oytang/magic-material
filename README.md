# magic-material

An integrated framework for materials property prediction and inverse design. This is a structured solution for one of the Physical Science Challenges of [ai4science hackathon](https://ai4science.io/), organized by [DeepVerse](deepverse.tech/en/), [Bota Bio](www.bota.bio), [Chemical.AI](https://chemical.ai/) and [Foreseen Biotechnology](www.foreseepharma.com/en-us).

## Overview

Generally, we provided an array of solutions for various tasks in the scope of Materials Science.

- **Property prediction** - Given the composition of a certain material, predict the corresponding property.
    - We illustrated our solutions with a wide range of data-driven models, including **classical machine learning models** (such as random forest and gradient boost) and latest **deep learning models** (such as Transformer and TabNet).
    - In addition, we constructed **stacking models** combining the above two kinds of models, to exploit the advantages of both models to greater extent.
    - We **handled missing values by** !!!!!!!!TODO!!!!!!!!!

- **Inverse design** - Given a set of desired properties, predict the composition of a certain material (usually, in a limited material design space, such as the 12-element space in this toy alloy data example), that has the cloest set of properties to the desire one.
    - We approched initially with **direct inverse modeling**, assuming an one-to-one mapping between compositions and properties, which is not necessarily true physically.
    - We developed an end-to-end pipeline of **inverse design using iterative optimization-based methods**. This approach best mimics the rational material design happening in wet laboratories, where scientists actively extract useful information from previous experimental results, try to design a better composition/formulation of materials, and test the new materials to validate the design.
    - **generative model**

## File Structure

    .
    ├── data                           # dataset, data analysis & preprocessing notebooks
    |   └── preprocessed               # preprocessed dataset
    ├── inverse_design_direct          # plain direct multi-layer perceptron models
    ├── inverse_design_generative      # conditional generative models
    ├── inverse_design_iterative       # iterative rational sampling (BO & PSO)
    ├── property_prediction_DL_NN      # deep learning models (Transformer, TabNet, etc.)
    ├── property_prediction_ML_tree    # classical machine learning models (random forest, gradient boost, etc.)
    ├── property_prediction_tree_NN    # stacking models combining tree models with NN models
    ├── DeepVerse_Challenge_1.ipynb    # example notebook provided by the organizer
    ├── LICENSE
    └── README.md

#  中文简介

材料性能预测与逆设计的集成框架。这是为[ai4science 黑客马拉松](https://ai4science.io/)的物理科学挑战之一提供的结构化解决方案，由[DeepVerse](deepverse.tech/en/)、[Bota Bio](www.bota.bio)、[Chemical.AI](https://chemical.ai/)和[Foreseen Biotechnology](www.foreseepharma.com/en-us)组织。

## 概述

总体上，我们为材料科学领域的各种任务提供了一系列的解决方案。

- **性能预测** - 给定某一材料的组成，预测其相应的性能。
    - 我们用一系列数据驱动模型阐述了我们的解决方案，包括**经典的机器学习模型**(如随机森林和梯度提升)和业界前沿的**深度学习模型**(如Transformer和TabNet)。
    - 此外，我们结合上述两种模型构建了**模型融合**，以更大程度地发挥两种模型的优势。
    - **处理缺失值** **！！！TODO！！！**

- **逆向设计** - 给定一组期望的性能，预测某种材料的组成 (通常，在有限的材料设计空间，例如本赛题示例中的12元素空间)，即具有最接近于期望性能的一组化学组分。
    - 我们最初采用的是**直接逆建模**，假设在组合和属性之间存在一对一的映射，这在物理上不一定是正确的。
    - 我们使用**基于迭代优化的方法**开发了一个端到端的逆向设计管道。这种方法最好地模仿了实验室中发生的合理材料设计，科学家能够很好地从以前的实验结果中提取有用的信息，尝试设计更好的材料组成/配方，并测试新材料来验证设计。
    - **生成模型** **！！！TODO！！！**

## 文件结构
    .
    ├── data                           # dataset, data analysis & preprocessing notebooks
    |   └── preprocessed               # preprocessed dataset
    ├── inverse_design_direct          # plain direct multi-layer perceptron models
    ├── inverse_design_generative      # conditional generative models
    ├── inverse_design_iterative       # iterative rational sampling (BO & PSO)
    ├── property_prediction_DL_NN      # deep learning models (Transformer, TabNet, etc.)
    ├── property_prediction_ML_tree    # classical machine learning models (random forest, gradient boost, etc.)
    ├── property_prediction_tree_NN    # stacking models combining tree models with NN models
    ├── DeepVerse_Challenge_1.ipynb    # example notebook provided by the organizer
    ├── LICENSE
    └── README.md
