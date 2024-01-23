export CUDA_VISIBLE_DEVICES=1
checkpoints="/opt/data/private/xwj/ML/LTSF-Linear-main/LSTM5_cpkts"

# multivariate setting of vanilla transformer
python -u run_longExp.py \
  --is_training 1 \
  --checkpoints $checkpoints \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model Transformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --freq 't' \
  --itr 5 \
  --batch_size 32 > log/ETTh1_96_96.log

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_192 \
#   --model Transformer \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --freq 't' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_336 \
#   --model Transformer \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --freq 't' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_720 \
#   --model Transformer \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --freq 't' \
#   --itr 1