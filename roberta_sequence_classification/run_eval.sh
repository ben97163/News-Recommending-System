# eval
python drive/MyDrive/ADL-final/run_classification.py \
  --model_name_or_path drive/MyDrive/ADL-final/result_with_content \
  --train_file data/train.json \
  --validation_file data/val.json \
  --metric_name accuracy \
  --text_column_names title,content \
  --label_column_name label \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --output_dir drive/MyDrive/ADL-final/eval_result/ \
  --overwrite_output_dir