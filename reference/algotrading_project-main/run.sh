if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

pred_len=4
model_name=NLinear

for seq_len in 25
do
  python -u run_backtest.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path data.csv\
    --model_id vix_"$seq_len"_"$pred_len" \
    --model "$model_name" \
    --data custom \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 12 \
    --des 'Exp' \
    --itr 1 --batch_size 16 --learning_rate 0.005\
    > logs/LongForecasting/"Final_NLinear_SPY_FINAL".log
done