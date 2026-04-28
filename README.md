# AgriExplain: Multimodal and Multitask Guava Classification with XAI and Agentic Multi-LLM Advisory Engine

## Overview

AgriExplain is an advanced end-to-end agricultural artificial intelligence framework developed for automated guava quality assessment. The system integrates multimodal learning, multitask classification, Explainable Artificial Intelligence (XAI), and Agentic Multi-LLM reasoning to identify guava maturity stage and disease condition while generating actionable recommendations for farmers, quality inspectors, and supply-chain stakeholders. Unlike conventional image classification systems that only provide predicted labels, AgriExplain delivers transparent explanations and practical guidance such as harvest timing, storage planning, disease isolation, and treatment recommendations.

## Objective of the Framework

The primary objective of AgriExplain is to simultaneously classify guava maturity and disease conditions through a unified intelligent model. For maturity analysis, the system predicts whether the fruit is Mature, Semi-Mature, or Immature. For disease assessment, it identifies Anthracnose, Scab, Styler Rot End, or Healthy fruits. The framework also aims to improve predictive accuracy using multimodal data sources, enhance computational efficiency through multitask learning, build trust through explainability, and provide real-world advisory outputs using agentic large language models.

## Dataset Modalities

AgriExplain uses multiple visual modalities to improve decision quality. RGB images are used to analyze peel color, surface texture, and shape information, which are important indicators of fruit maturity. Thermal images help capture internal physiological changes and temperature distribution patterns associated with ripening. Disease images are used to identify lesion patterns, discoloration, decay regions, and visible infection symptoms. By combining these complementary modalities, the framework obtains a richer understanding of fruit quality than any single data source alone.

## Data Preprocessing

Before model training, all input images undergo a preprocessing pipeline to ensure consistency and quality. Images are resized to a fixed resolution suitable for deep learning models. Pixel values are normalized for stable optimization during training. Data augmentation techniques such as flipping, rotation, and brightness variation are applied to improve generalization and robustness. Noise removal and quality checks are also performed to minimize irrelevant artifacts and improve learning performance.

## Deep Learning Backbone Benchmarking

To construct a reliable perception layer, AgriExplain benchmarks multiple state-of-the-art convolutional neural network architectures including ResNet50, DenseNet121, and EfficientNetB0. These models are evaluated as feature extractors rather than standalone classifiers. ResNet50 is effective for learning deep hierarchical representations, DenseNet121 promotes feature reuse and better gradient flow, while EfficientNetB0 offers lightweight and fast inference suitable for edge deployment. The models are compared using Accuracy, Precision, Recall, F1-score, and inference speed. The best-performing backbone is selected for downstream multimodal fusion and multitask prediction.

## Multimodal Fusion Module

A major challenge in fruit assessment is that surface appearance alone may not fully represent internal maturity. To address this, AgriExplain integrates RGB and thermal information through a multimodal fusion module. Early fusion combines features at initial layers, allowing joint low-level representation learning. Late fusion processes each modality independently and combines predictions at the decision level. Attention-based fusion dynamically assigns importance weights to modalities depending on the sample. This module improves maturity classification by leveraging complementary external and internal fruit cues.

## Multitask Learning Module

Instead of using separate models for maturity and disease detection, AgriExplain employs multitask learning to solve both tasks simultaneously. A shared backbone extracts common features such as texture, color transitions, lesion structures, and shape patterns. These shared embeddings are then passed to two task-specific heads: one for maturity classification and another for disease recognition. This strategy reduces memory usage, lowers inference time, improves generalization, and enables efficient deployment in practical agricultural environments.

## Explainable AI Pipeline

To improve transparency and user trust, AgriExplain integrates an Explainable AI pipeline. Deep learning systems are often considered black-box models, which can limit adoption in real-world farming scenarios. Therefore, the framework uses Grad-CAM, Integrated Gradients, and Occlusion Sensitivity to visually explain predictions. Grad-CAM highlights important image regions responsible for decisions, Integrated Gradients provides pixel-level attribution scores, and Occlusion Sensitivity measures the impact of masking image areas. These explanations help confirm whether the model focuses on biologically meaningful cues such as disease lesions, peel discoloration, thermal hotspots, or ripening zones rather than irrelevant background information.

## LangGraph Agentic Multi-LLM Advisory Engine

One of the most innovative components of AgriExplain is its LangGraph-based Agentic Multi-LLM Advisory Engine. Instead of ending with classification results, the framework transforms predictions into agricultural intelligence. The advisory engine receives maturity prediction, disease category, confidence scores, and XAI summaries as structured inputs. These signals are then processed by locally hosted language models such as Llama 3 and Qwen 2.5, enabling full offline deployment, lower operational cost, and stronger data privacy.

Using LangGraph orchestration, multiple specialized agents collaborate to generate final recommendations. A Diagnosis Agent interprets the predicted fruit condition, an Explanation Agent summarizes XAI evidence, an Advisory Agent provides practical guidance, a Verification Agent checks hallucinations or unsafe responses, and a Judge Agent selects the most useful final output. This architecture ensures more reliable and transparent responses than a single-model chatbot approach.

## Example Advisory Output

For example, if the system predicts that a guava is Mature and infected with Anthracnose at high confidence, the advisory engine may recommend immediate separation from healthy fruits, avoidance of long-term storage, inspection of neighboring fruits, and prioritization for rapid sale if quality standards permit. In this way, AgriExplain converts raw predictions into actionable operational decisions.

## Fallback Reliability Layer

To ensure uninterrupted deployment, the framework includes rule-based fallback templates when language models are unavailable or fail quality checks. For instance, a Mature and Healthy fruit may trigger a “Ready for harvest” recommendation, while a diseased fruit may generate a “Separate and inspect batch” alert. This reliability layer ensures consistent performance in low-resource or offline environments.

## Practical Applications

AgriExplain can be deployed across multiple smart agriculture scenarios including farms, post-harvest grading centers, cold-storage systems, fruit supply chains, and quality inspection facilities. It can reduce dependency on manual inspection, improve grading consistency, support faster decision-making, and enhance food quality management.

## Technologies Used

The framework can be implemented using Python, TensorFlow or PyTorch, OpenCV, Scikit-learn, Grad-CAM libraries, LangGraph, Ollama or other local LLM serving systems, and optional Streamlit interfaces for deployment.

## Future Scope

Future improvements may include multilingual farmer advisory systems, voice-based recommendations, mobile deployment, real-time camera grading, shelf-life prediction, and Retrieval-Augmented Generation (RAG) integration using agricultural knowledge bases.

## Conclusion

AgriExplain is more than a guava classification model. It is a trustworthy agricultural intelligence ecosystem that combines multimodal perception, multitask learning, explainability, and agentic reasoning into a single framework. By converting visual predictions into understandable and actionable guidance, AgriExplain represents the next generation of explainable smart farming systems.
