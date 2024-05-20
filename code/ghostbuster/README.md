# Ghostbuster instructions


1. First get the logprobs for all of your texts using two models with the same tokenizers. I used tinyllama and llama7b. These should be saved in a directory where each text has a separate file. The format for the files is `token logprobs\ntoken logprobs` and so on.

2. Then run `run.py` to get the features for the texts.
3. Finally, run `train_lr.py` to train the model on the features. Below is the command used to run the script.

```sh
python train_lr.py \
--feature_path "../custom_7b-tl-ft-ignore-25-cmin4-cmax4-m20" \
--model_type "vote" \
--C 100 \
--train_on_all_data
```

Note, use `--train_on_all_data` only after finding a good value of C