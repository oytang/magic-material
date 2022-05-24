# magic-material

An integrated framework for materials property prediction and inverse design. This is a structured solution for one of the Physical Science Challenges of [ai4science hackathon](https://ai4science.io/), organized by [DeepVerse](deepverse.tech/en/), [Bota Bio](www.bota.bio), [Chemical.AI](https://chemical.ai/) and [Foreseen Biotechnology](www.foreseepharma.com/en-us).

## Overview

Generally, we provided an array of solutions for various tasks in the scope of Materials Science.

- **Property prediction** - Given the composition of a certain material, predict the corresponding property.
    - We illustrated our solutions with a wide range of data-driven models, including **classical machine learning models** (such as random forest and gradient boost) and latest **deep learning models** (such as Transformer and TabNet).
    - In addition, we constructed **ensemble models** combining the above two kinds of models, to exploit the advantages of both models to greater extent.
    - We **handled missing values by estimation** of them using the mean of that value in the rest of the dataset. The largest advantage of this apporoach is that we can directly translate developed pipeline from complete dataset to incomplete dataset. The obvious disadvantage is that incorrect estimation can bring large bias to the model.

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
    ├── property_prediction_tree_NN    # ensemble models combining tree models with NN models
    ├── DeepVerse_Challenge_1.ipynb    # example notebook provided by the organizer
    ├── LICENSE
    └── README.md

## TODO: 双语对照
