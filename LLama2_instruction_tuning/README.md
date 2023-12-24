# Training
## Set up
### Environment
- Addition to the packages provided by TAs in the slides. You also need to install the following packages
```
pip install scipy datasets accelerate
```
### Training Scripts
- You can train the model by the following command
```
python train.py --model_path {your llama2 model directory} --train_file_path {your train.json file path} --test_file_path {your test.json file path} --output_dir {output save directory path} --batch_size {your batch size}
```
- To reproduce my result, you need to set batch size to 4 (this may cause OOM in your environment)
