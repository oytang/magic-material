<a href="url"><img src="MagiXterial-logos.jpeg" align="left" height="150" width="150" ></a>

# magic-material

An integrated framework for materials property prediction and inverse design, provided by team **MagiXterial**. This is a structured solution for one of the Physical Science Challenges of [ai4science hackathon](https://ai4science.io/), organized by [DeepVerse](deepverse.tech/en/), [Bota Bio](www.bota.bio), [Chemical.AI](https://chemical.ai/) and [Foreseen Biotechnology](www.foreseepharma.com/en-us).

## Overview

Generally, we provided an array of solutions for various tasks in the scope of Materials Science.

- **Property prediction** - Given the composition of a certain material, predict the corresponding property.
    - We illustrated our solutions with a wide range of data-driven models, including **classical machine learning models** (such as random forest and gradient boost) and latest **deep learning models** (such as Transformer and TabNet).
    - In addition, we constructed **stacking models** combining the above two kinds of models, to exploit the advantages of both models to greater extent.
    - We **handled missing data** according to how data are missing in a sample:
        - If there is only one missing value in all elements,  that missing value can be inferred/calculated from the other values following the rule that the summation of elements is one (in real cases this can be leveraging conservation of mass or other physical rules)
        - If there are more than 3 missing values in composition (> 1/4), then the quality of that data is not very good and we choose to drop that data point.
        - If there are missing values in properties, we drop that data point as well. Because even if we can give the prediction or fill other values but we don’t want the model to be established with inaccurate label for a supervised ML problem.
        - For the rest of data which contains the missing value, we fill it with -1. 

- **Inverse design** - Given a set of desired properties, predict the composition of a certain material (usually, in a limited material design space, such as the 12-element space in this toy alloy data example), that has the cloest set of properties to the desire one.
    - We approched initially with **direct inverse modeling**, assuming an one-to-one mapping between compositions and properties, which is not necessarily true physically.
    - We developed an end-to-end pipeline of **inverse design using iterative optimization-based methods**. This approach best mimics the rational material design happening in wet laboratories, where scientists actively extract useful information from previous experimental results, try to design a better composition/formulation of materials, and test the new materials to validate the design.
    - Further, in the face of larger data, we use the more general **generation model** as the reverse design method, map the joint vector of materials and properties into a continuous high-dimensional space through CVAE method, and obtain the generation of material formula that meets the substitution conditions

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
    ├── run_inverse_design.ipynb       # an inverse-design demo
    ├── LICENSE
    └── README.md

#  中文简介

材料性能预测与逆设计的集成框架。这是为[ai4science 黑客马拉松](https://ai4science.io/)的物理科学挑战之一提供的结构化解决方案，由[DeepVerse](deepverse.tech/en/)、[Bota Bio](www.bota.bio)、[Chemical.AI](https://chemical.ai/)和[Foreseen Biotechnology](www.foreseepharma.com/en-us)组织。

## 概述

总体上，我们为材料科学领域的各种任务提供了一系列的解决方案。

- **性能预测** - 给定某一材料的组成，预测其相应的性能。
    - 我们用一系列数据驱动模型阐述了我们的解决方案，包括**经典的机器学习模型**(如随机森林和梯度提升)和业界前沿的**深度学习模型**(如Transformer和TabNet)。
    - 此外，我们结合上述两种模型构建了**模型融合**，以更大程度地发挥两种模型的优势。
    - 我们根据样本中数据缺失的具体情况来**处理缺失的数据**。
        - 如果所有元素中只有一个缺失值，那么这个缺失值可以按照元素之和为1的规则从其他值中推断/计算出来（在实际案例中，这相当于可以利用质量守恒或其他物理规则）。
        - 如果成分中有超过3个缺失值（>1/4），那么该数据的质量就不是很好，我们选择放弃该数据点。
        - 如果属性中存在缺失值，我们也会放弃这个数据点。因为即使我们可以给出预测或填补其他值，但我们不希望在有监督机器学习问题上用不准确的标签建立模型。
        - 对于剩下的包含缺失值的数据，我们用-1来填补它。

- **逆向设计** - 给定一组期望的性能，预测某种材料的组成 (通常，在有限的材料设计空间，例如本赛题示例中的12元素空间)，即具有最接近于期望性能的一组化学组分。
    - 我们最初采用的是**直接逆建模**，假设在组合和属性之间存在一对一的映射，这在物理上不一定是正确的。
    - 我们使用**基于迭代优化的方法**开发了一个端到端的逆向设计管道。这种方法最好地模仿了实验室中发生的合理材料设计，科学家能够很好地从以前的实验结果中提取有用的信息，尝试设计更好的材料组成/配方，并测试新材料来验证设计。
    - 进一步说，面对更大的数据，我们用更通用的**生成模型**作为反向设计方法，通过条件变分解码器(CVAE)方法将材料和性能的联合向量映射到连续的高维空间，得到满足给定条件的材料配方的生成。

## 文件结构
    .
    ├── data                           # 数据集、数据分析和预处理笔记本
    |   └── preprocessed               # 预处理过的数据集
    ├── inverse_design_direct          # 直接多层感知器模型
    ├── inverse_design_generative      # 条件生成模型
    ├── inverse_design_iterative       # 迭代优化采样（贝叶斯优化、粒子群优化）
    ├── property_prediction_DL_NN      # 深度学习模型（Transformer，TabNet，等）
    ├── property_prediction_ML_tree    # 经典机器学习模型（随机森林，梯度提升，等）
    ├── property_prediction_tree_NN    # 结合树状模型和神经网络模型的堆叠模型
    ├── DeepVerse_Challenge_1.ipynb    # 主办方提供的笔记本范例
    ├── run_inverse_design.ipynb       # 逆向设计模型使用展示
    ├── LICENSE
    └── README.md
