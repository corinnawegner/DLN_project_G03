# DNLP SS24 Final Project

This is the starting code for the default final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. You can find the handout [here](https://docs.google.com/document/d/1pZiPDbcUVhU9ODeMUI_lXZKQWSsxr7GO/edit?usp=sharing&ouid=112211987267179322743&rtpof=true&sd=true)

In this project, you will implement some important components of the BERT model to better understanding its architecture.
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh` (Use `setup_gwdg.sh` if you are using the GWDG clusters).
* Libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).
* Use this template to create your README file of your repository: <https://github.com/gipplab/dnlp_readme_template>

## Project Description

Please refer to the project description for a through explanation of the project and its parts.

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

For the 2024 edition of the DNLP course at the University of Göttingen, the project was modified by [Niklas Bauer](https://github.com/ItsNiklas/), [Jonas Lührs](https://github.com/JonasLuehrs), ...


Group Name
Here, you should provide a group name you want to be referred to as, as well as the names (and optionally contact info) of all group members. Additionally, define a group leader, i.e. one person in your group who is the primary contact if tutors want to communicate with your group. Also, mention here which tutor is assigned to your project.

Example:

Group name:

Group code:

Group repository:

Tutor responsible:

Group team leader:

Group members:

Setup instructions
Explain how we can run your code in this section. We should be able to reproduce the results you've obtained.

In addition, if you used libraries that were not included in the conda environment 'dnlp' explain the exact installation instructions or provide a .sh file for the installation.

Which files do we have to execute to train/evaluate your models? Write down the command which you used to execute the experiments. We should be able to reproduce the experiments/results.

Hint: At the end of the project you can set up a new environment and follow your setup instructions making sure they are sufficient and if you can reproduce your results.

Methodology
In this section explain what and how you did your project.

If you are unsure how this is done, check any research paper. They all describe their methods/processes. Describe briefly the ideas that you implemented to improve the model. Make sure to indicate how are you using existing ideas and extending them. We should be able to understand your project's contribution.

Experiments
Keep track of your experiments here. What are the experiments? Which tasks and models are you considering?

Write down all the main experiments and results you did, even if they didn't yield an improved performance. Bad results are also results. The main findings/trends should be discussed properly. Why a specific model was better/worse than the other?

You are required to implement one baseline and improvement per task. Of course, you can include more experiments/improvements and discuss them.

You are free to include other metrics in your evaluation to have a more complete discussion.

Be creative and ambitious.

For each experiment answer briefly the questions:

What experiments are you executing? Don't forget to tell how you are evaluating things.
What were your expectations for this experiment?
What have you changed compared to the base model (or to previous experiments, if you run experiments on top of each other)?
What were the results?
Add relevant metrics and plots that describe the outcome of the experiment well.
Discuss the results. Why did improvement A perform better/worse compared to other improvements? Did the outcome match your expectations? Can you recognize any trends or patterns?
Results
Summarize all the results of your experiments in tables:

Stanford Sentiment Treebank (SST)	Metric 1	Metric n
Baseline	45.23%	...
Improvement 1	58.56%	...
Improvement 2	52.11%	...
...	...	...
Quora Question Pairs (QQP)	Metric 1	Metric n
Baseline	45.23%	...
Improvement 1	58.56%	...
Improvement 2	52.11%	...
...	...	...
Semantic Textual Similarity (STS)	Metric 1	Metric n
Baseline	45.23%	...
Improvement 1	58.56%	...
Improvement 2	52.11%	...
...	...	...
Paraphrase Type Detection (PTD)	Metric 1	Metric n
Baseline	45.23%	...
Improvement 1	58.56%	...
Improvement 2	52.11%	...
...	...	...
Paraphrase Type Generation (PTG)	Metric 1	Metric n
Baseline	45.23%	...
Improvement 1	58.56%	...
Improvement 2	52.11%	...
...	...	...
Discuss your results, observations, correlations, etc.

Results should have three-digit precision.

Hyperparameter Optimization
Describe briefly how you found your optimal hyperparameter. If you focussed strongly on Hyperparameter Optimization, you can also include it in the Experiment section.

Note: Random parameter optimization with no motivation/discussion is not interesting and will be graded accordingly

Visualizations
Add relevant graphs of your experiments here. Those graphs should show relevant metrics (accuracy, validation loss, etc.) during the training. Compare the different training processes of your improvements in those graphs.

For example, you could analyze different questions with those plots like:

Does improvement A converge faster during training than improvement B?
Does Improvement B converge slower but perform better in the end?
etc...
Members Contribution
Explain what member did what in the project:

Member 1: implemented the training objective using X, Y, and Z. Supported member 2 in refactoring the code. Data cleaning, etc.

Member 2: ...

...

We should be able to understand each member's contribution within 5 minutes.

AI-Usage Card
Artificial Intelligence (AI) aided the development of this project. Please add a link to your AI-Usage card here.

References
Write down all your references (other repositories, papers, etc.) that you used for your project.
