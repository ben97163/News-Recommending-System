llama_path=$1
peft_path=$2
input_path=$3
output_path=$4

python predict.py --base_model_path $llama_path --peft_path $peft_path --test_data_path $input_path --output_path $output_path