# Fake News Detection

In this repository I fine-tune several BERT models and GPT-2 for the task of fake news detection (binary task). Bert-base proved to be the best model (89% accuracy and ~90% precision recall and F1-score). Also, GPT-2 despite of being trained for next word prediction, turned out to be have a very high accuracy (87%). This can be explained by two reasons
1) the fact that  I used the last token of the last hidden state as an equivalent of the BERT's CLS token, (although it is not the same) with the intuition that since masked attention is used, this token will be the only one capturing the information of the whole sequence.
2) I also used mean pooling (the mean of the embeddings of all the other tokens).
After concatenation of 1 and 2, I passed this through a linear transformation and sigmoid activation to get my final output.
