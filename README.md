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

Before running the code, you need to put the downloaded dataset into the corresponding folder under the `/data` path.

## Usage

For simplicity, here we take Pearl as an example：

### Fine-Tune 
Before processing the data, we first fine-tune the retrieval model. We need it to get the embedding of the data. The fine-tuned dataset can be found in the `/data/fine_tune_data` folder.
```
cd data/script
python fine_tuning.py
```

### Preprocessed

Use the following commands to process Pearl dataset into a unified format. 

```
python pearl_pdata.py
```

You can find the preprocessed data in the `/data/pearl` folder.

### Run

To run **ToT-GRR(LLaMA,BFS)** on Pearl (with default hyper-parameters):

```
cd ../../
python main.py --dataset=pearl --base_model=Llama --search=bfs
```

You can run the **LLaMA** baseline model in Pearl using the following commands:
```
python main.py --dataset=pearl --baseline=Llama
```

>[!NOTE]
>If you are using the GPT-3.5-turbo model, you need to fill in the **API key** in the ChatGPT class definition in `base_models.py` in advance.

