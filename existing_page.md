2024-08-30

# Personalized Daily Arxiv Papers 08/29/2024
Total relevant papers: 4

Paper selection prompt and criteria at the bottom

Table of contents with paper titles:

0. [Boosting Lossless Speculative Decoding via Feature Sampling and Partial Alignment Distillation](#link0)
**Authors:** Lujun Gui, Bin Xiao, Lei Su, Weipeng Chen

1. [Implicit Geometry of Next-token Prediction: From Language Sparsity Patterns to Model Representations](#link1)
**Authors:** Yize Zhao, Tina Behnia, Vala Vakilian, Christos Thrampoulidis

2. [GANs Conditioning Methods: A Survey](#link2)
**Authors:** Anis Bourou, Auguste Genovesio, Val\'erie Mezger

3. [Avoiding Generative Model Writer's Block With Embedding Nudging](#link3)
**Authors:** Ali Zand, Milad Nasr

---
## 0. [Boosting Lossless Speculative Decoding via Feature Sampling and Partial Alignment Distillation](https://arxiv.org/abs/2408.15562) <a id="link0"></a>
**ArXiv ID:** 2408.15562
**Authors:** Lujun Gui, Bin Xiao, Lei Su, Weipeng Chen

**Abstract:** arXiv:2408.15562v1 Announce Type: new  Abstract: Lossless speculative decoding accelerates target large language model (LLM) inference by employing a lightweight draft model for generating tree-structured candidates, which are subsequently verified in parallel by the target LLM. Currently, effective approaches leverage feature-level rather than token-level autoregression within the draft model to facilitate more straightforward predictions and enhanced knowledge distillation. In this paper, we reassess these approaches and propose FSPAD (Feature Sampling and Partial Alignment Distillation for Lossless Speculative Decoding), which introduces two straightforward and effective components within the existing framework to boost lossless speculative decoding. Firstly, FSPAD utilizes token embeddings to sample features of the target LLM in high-dimensional space before feeding them into the draft model, due to the inherent uncertainty of the features preventing the draft model from obtaining the specific token output by the target LLM. Secondly, FSPAD introduces partial alignment distillation to weaken the draft model's connection between features and logits, aiming to reduce the conflict between feature alignment and logit confidence during training. Our experiments include both greedy and non-greedy decoding on the largest and smallest models from the Vicuna and LLaMA3-Instruct series, as well as tasks in multi-turn conversation, translation, summarization, question answering, mathematical reasoning, and retrieval-augmented generation. The results show that FSPAD outperforms the state-of-the-art method across all the aforementioned tasks and target LLMs.

**Comment:** Matches criterion 1 as it discusses knowledge distillation methods from a Teacher to a Student model.
**Relevance:** 8
**Novelty:** 6

---

## 1. [Implicit Geometry of Next-token Prediction: From Language Sparsity Patterns to Model Representations](https://arxiv.org/abs/2408.15417) <a id="link1"></a>
**ArXiv ID:** 2408.15417
**Authors:** Yize Zhao, Tina Behnia, Vala Vakilian, Christos Thrampoulidis

**Abstract:** arXiv:2408.15417v1 Announce Type: new  Abstract: Next-token prediction (NTP) over large text corpora has become the go-to paradigm to train large language models. Yet, it remains unclear how NTP influences the mapping of linguistic patterns to geometric properties of the resulting model representations. We frame training of large language models as soft-label classification over sparse probabilistic label vectors, coupled with an analytical approximation that allows unrestricted generation of context embeddings. This approach links NTP training to rank-constrained, nuclear-norm regularized optimization in the logit domain, offering a framework for analyzing the geometry of word and context embeddings. In large embedding spaces, we find that NTP implicitly favors learning logits with a sparse plus low-rank structure. While the sparse component captures the co-occurrence frequency of context-word pairs, the orthogonal low-rank component, which becomes dominant as training progresses, depends solely on the sparsity pattern of the co-occurrence matrix. Consequently, when projected onto an appropriate subspace, representations of contexts that are followed by the same set of next-tokens collapse, a phenomenon we term subspace-collapse. We validate our findings on synthetic and small-scale real language datasets. Finally, we outline potential research directions aimed at deepening the understanding of NTP's influence on the learning of linguistic patterns and regularities.

**Comment:** Relevant to criterion 4 as it discusses the geometry of model representations in the context of language models, which could be related to scaling laws.
**Relevance:** 5
**Novelty:** 6

---

## 2. [GANs Conditioning Methods: A Survey](https://arxiv.org/abs/2408.15640) <a id="link2"></a>
**ArXiv ID:** 2408.15640
**Authors:** Anis Bourou, Auguste Genovesio, Val\'erie Mezger

**Abstract:** arXiv:2408.15640v1 Announce Type: new  Abstract: In recent years, Generative Adversarial Networks (GANs) have seen significant advancements, leading to their widespread adoption across various fields. The original GAN architecture enables the generation of images without any specific control over the content, making it an unconditional generation process. However, many practical applications require precise control over the generated output, which has led to the development of conditional GANs (cGANs) that incorporate explicit conditioning to guide the generation process. cGANs extend the original framework by incorporating additional information (conditions), enabling the generation of samples that adhere to that specific criteria. Various conditioning methods have been proposed, each differing in how they integrate the conditioning information into both the generator and the discriminator networks. In this work, we review the conditioning methods proposed for GANs, exploring the characteristics of each method and highlighting their unique mechanisms and theoretical foundations. Furthermore, we conduct a comparative analysis of these methods, evaluating their performance on various image datasets. Through these analyses, we aim to provide insights into the strengths and limitations of various conditioning techniques, guiding future research and application in generative modeling.

**Comment:** Relevant to criterion 5 as it surveys GANs conditioning methods which could be applied to improve recommender systems.
**Relevance:** 5
**Novelty:** 4

---

## 3. [Avoiding Generative Model Writer's Block With Embedding Nudging](https://arxiv.org/abs/2408.15450) <a id="link3"></a>
**ArXiv ID:** 2408.15450
**Authors:** Ali Zand, Milad Nasr

**Abstract:** arXiv:2408.15450v1 Announce Type: new  Abstract: Generative image models, since introduction, have become a global phenomenon. From new arts becoming possible to new vectors of abuse, many new capabilities have become available. One of the challenging issues with generative models is controlling the generation process specially to prevent specific generations classes or instances . There are several reasons why one may want to control the output of generative models, ranging from privacy and safety concerns to application limitations or user preferences   To address memorization and privacy challenges, there has been considerable research dedicated to filtering prompts or filtering the outputs of these models. What all these solutions have in common is that at the end of the day they stop the model from producing anything, hence limiting the usability of the model. In this paper, we propose a method for addressing this usability issue by making it possible to steer away from unwanted concepts (when detected in model's output) and still generating outputs. In particular we focus on the latent diffusion image generative models and how one can prevent them to generate particular images while generating similar images with limited overhead.   We focus on mitigating issues like image memorization, demonstrating our technique's effectiveness through qualitative and quantitative evaluations. Our method successfully prevents the generation of memorized training images while maintaining comparable image quality and relevance to the unmodified model.

**Comment:** Not relevant to the specified criteria but discusses controlling the output of generative models, which could be of general interest.
**Relevance:** 3
**Novelty:** 5

---


---

## Paper selection prompt
 1. Studies knowledge distillation methods from a Teacher to a Student model
    - Relevant: Papers that introduce easily employable methods that do not require a completely new architecture, such as new loss functions or training schemes
    - Not Relevant: Papers that have a student teacher model but it is not the main focus of the paper and they do not develop a novel technique for that task
 2. Studies Uncertainty estimation, Deep evidential uncertainty, Bayesian uncertainty for neural networks
    - Relevant: Any paper that tries to estimate the epistemic and aleatoric uncertainty of neural network predictions.
    - Not Relevant: Methods which require a lot extra overhead which would make it impractical for inference
 3. Studies Named Entity Extraction using neural networks and in particular the dataset curation process using active learning and human in the loop
    - Relevant: Papers which generate training data with LLMS but have a way to either detect errors automatically and direct humans labelers to look at them or have a way to somehow filter out the bad labels.
    - Not Relevant: Papers which do entity extraction using only human labels
 4. Studies 'scaling laws' in the context of neural networks. Scaling laws refer to the very clear power-law relationship between the size or computational power used to train a model and the performance of that model.
    - Relevant: theoretical or conceptual explanation behind scaling laws for language models.
    - Not relevant: papers that have experiments at different model scales (but do not explicitly fit a scaling law) or papers that mention scaling laws, but the scaling laws are not the central subject of the paper
 5. Papers that study recommender systems using neural networks and propose new methods to improve these
    - Relevant: Papers that could allow to improve an existing recommender system, either through data curation, model improvements, training methods etc...
    - Not Relevant: vague papers that do not provide any reusability such as simply using a pivate dataset and providing any new architecture component that could be reused

 In suggesting papers to your friend, remember that he enjoys papers on statistical machine learning, and generative modeling in natural language processing.
 Your friend also likes learning about surprising empirical results in language models, as well as clever statistical tricks.
 He does not want to read papers that are about primarily applications of methods to specific domains.
