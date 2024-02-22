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

# Github directory structure

```bash
ADL_final
├── LLama2_classification
│   ├── inference.py
│   ├── train.py
│   └── utils.py
├── LLama2_instruction_tuning
│   ├── README.md
│   ├── draw.py
│   ├── ppl.py
│   ├── predict.py
│   ├── requirement.txt
│   ├── run.sh
│   ├── train.py
│   └── utils.py
├── Llama2_Regression
│   ├── eval.py
│   ├── inference_demo.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── README.md
├── crawl
│   └── crawl.py
├── data
│   ├── news_data.json
│   ├── total.json
│   ├── total_views.json
│   ├── train.json
│   └── val.json
├── demo
│   ├── crawl.py
│   ├── demo.py
│   ├── score.pt
│   └── utils.py
├── display.py
├── distribution.png
├── distribution2.png
├── roberta_sequence_classification
│   ├── classification.ipynb
│   ├── requirements.txt
│   ├── run_classification.py
│   ├── run_eval.sh
│   └── run_train.sh
└── utlis
    ├── define_class.py
    ├── get_views.py
    ├── split_class.py
    └── split_view.py
```
# Demo
```
cd demo
streamlit run demo.py --server.address=0.0.0.0
# click on the link: http://0.0.0.0:8501
```# News-Recommending-System
