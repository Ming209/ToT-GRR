# Towards Logic-Controllable Conversational Recommendation: A Tree-of-Thought-Guided Strategy with Large Language Models as Core Hub


## Overview
Conversational recommender systems (CRSs), as a widely used information system, aim to embed personalized recommendations into natural, multi-turn dialogue, enabling users to articulate and refine preferences interactively. However, despite recent advances with large language models (LLMs), most existing CRS approaches remain limited: their greedy, single-step generation often causes hallucinations, shallow reasoning, and poor alignment with evolving user preferences—resulting in degraded recommendation quality and diminished user satisfaction in real-world deployments.

We propose ToT-GRR (*T*ree-*o*f-*T*hought *G*uided *Retrieval and *R*ecommendation), a framework that reformulates CRS as a planning problem over dialogue space. ToT-GRR integrates tree-of-thought exploration, which generates and evaluates diverse reasoning paths with the aid of a user simulator, and a feedback-aware retrieval module, which grounds recommendations in dynamically updated candidate sets. This design enables the system to reason beyond immediate turns while remaining aligned with evolving user preferences.

Extensive experiments on two benchmark datasets show that ToT-GRR consistently outperforms strong LLM baselines in both success rate and dialogue efficiency. Ablation and sensitivity analyses confirm the complementary roles of retrieval and reasoning, while a case study illustrates how ToT-GRR produces accurate and contextually appropriate recommendations beyond direct prompting.

Taken together, these findings advance the design of conversational recommendation systems by showing how structured reasoning and feedback-aware retrieval can be combined to enhance personalization and adaptivity under uncertainty. Our study contributes to the information systems literature by providing a principled approach for building adaptive CRS, and offers practical insights for leveraging generative AI to improve recommendation quality, dialogue efficiency, and overall user satisfaction on digital platforms.



<!-- <div align="center">
  <a href="https://youtu.be/gYCeTO0fLvE">
    <img src="https://img.youtube.com/vi/gYCeTO0fLvE/hqdefault.jpg" alt="LANE DEMO">
  </a>
  <p style="font-weight: bold; font-size: 16px;">Click to play the video</p>
</div> -->

##  Installation

### Clone the repositoru

```
git clone https://github.com/Ming209/LANE.git
cd ToT-GRR
```


### Requirement

Use the following command to install the required dependencies:

```
pip install -r requirements.txt
```


### Datesets

You can download the two datasets we used from the link below.

- [Pearl](https://huggingface.co/datasets/LangAGI-Lab/pearl)

- [Redial](https://github.com/ReDialData/website/tree/data)

Before running the code, you need to put the downloaded dataset into the corresponding folder under the `data/dateset/raw` path.

## Usage

For simplicity, here we take Beauty as an example：

### Preprocessed

Use the following code to process different datasets into a unified txt format. You can find them in the `data/dataset/serialize` folder:

```
python data/script/serialize_Beauty.py
```

Then use the following code to process the txt data to get the preprocessed data.You can find them in the `data/preprocessed` folder:

```
python Datapreprocessed.py --dataset=Beauty
```

>[!NOTE]
>the GPT-3.5-turbo API will be called here, and you need to fill in the **API key** in the *gpt_request()* function in `utils.py` in advance


### Train

To train **Baseline(SASRec)** on Beauty (with default hyper-parameters):

```
python train_baseline.py --dataset=Beauty --model=SASRec --config_path=config/sasrec.json --maxlen=50
```


To train **LANE(LANE-SASRec)** on Beauty (with default hyper-parameters):

```
python train_LANE.py --dataset=Beauty --inte_model=SASRec --inte_model_config_path=config/sasrec.json --maxlen=50
```

You can add the `--generate_explanation` parameter to let **LANE** generate explainable recommendations, which also requires calling the GPT API. Considering the call cost, we only randomly select 100 samples by default to generate explainable recommendations.


