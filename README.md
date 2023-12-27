# News Crawling
For preparing news data
```
python crawl/crawl.py 
```

# Taiwan-LLaMa
## Regression experiment
For model training
```
python Llama2_Regression/train.py --train_data data/train.json --eval_data data/val.json --epoch 2 
```

For inference
```
python Llama2_Regression/predict.py --peft_path "trained lora weight" --eval_data data/val.json 
```

For evaluation
```
python Llama2_Regression/eval.py --reference_file "ground truth data" --prediction_file "result from inferece"
```

## Classification experiment
For model training
```
python LLama2_classification/train.py --train_file "train_file" --test_file "test_file"
```

For inference
```
python LLama2_classification/inference.py
```

# RoBERTa
## Sentence Classification experiment
For model training
```
bash run_train.sh
```

For inference
```
bash run_eval.sh
```
refer to the ipynb file for reference
