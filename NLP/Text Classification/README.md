## Overview:
Intent classification is a vital component of dialog systems. This notebook performs multi-class intent classification on a widely-known single banking domain intent dataset, [BANKING77-OOS](https://paperswithcode.com/dataset/banking77-oos), introduced by [Zhang et al. 2021] (https://arxiv.org/pdf/2106.04564v3.pdf).
As explained in the paper, BANKING77 originally contained 77 intents with no initial Out-Of-Scope (OOS) intents. The newly formed dataset BANKING77-OOS is composed of 50 in-scope intents while 27 intents are held out, to form OOS examples with. 

The aim of the paper is to study the performance and robustness of state-of-the-art Pretrained Transformers in detecting OOS intents (both in-domain such as requesting a banking service that is unsupported by the banking system in-hand vs. out-of-domain intents).

However, the purpose of this notebook is simply to fine-tune a pre-trained transformer model, RoBERTa, to accurately detect intents from the 50 intents in BANKING77-OOS, in a banking domain, using TPUs. 

unlike BERT, RoBERTa (Robustly Optimized BERT pre-training Approach) uses dynamic masking, instead of statically masking the same part of a sentence each epoch, which helps it learn more robust word representations.
Moreover, RoBERTa is trained on 10 times more textual data than BERT was originally. It's become a popular model choice in both research and industy 
serving as a base model for other NLP models and outperforming BERT in NLP tasks such as translation, classification and question answering.


The final model fine-tuned in this notebook reaches a **0.9102** top-1-accuracy on validation set, and a **0.9740** top-3-accuracy on the validation set. (Of course, to be more exact, one should actually split the data into train, test and validation and finally evaluate on the test dataset).
   

## Data:
Here is a sample of the BANKING77-OOS dataset below:
<img width="729" alt="Screenshot 2023-07-12 at 6 29 12 PM" src="https://github.com/faridashahata/Portfolio/assets/113303940/5dc372e8-9d4e-43af-aab7-2f24c93d7604">


The dataset consists of 9411 utterance-label pairs, which are further split up into train-val-test datasets. 
## Fine-tuning Regime(s):


We used the base configuration of RoBERTa, `roberta-base`.
The Adam optimizer was used, with gradient clipping, with a clipping value of 1.0. No linear warmup scheduling nor weight decay were applied. 

The metrics used to evaluate the model were categorical accuracy and top-3 categorical accuracy. Using Colab TPUs, we are able to set the batch size as high as 256.

For each model, we perform hyperparameters search for learning rate values ∈ {1e − 5, 2e − 5, 3e − 5}, and the number of training epochs ∈ {15, 20, 25, 30, 50, 60}.


## References: 

https://paperswithcode.com/dataset/banking77-oos

https://arxiv.org/pdf/2106.04564v3.pdf

https://towardsdatascience.com/news-category-classification-fine-tuning-roberta-on-tpus-with-tensorflow-f057c37b093

     
