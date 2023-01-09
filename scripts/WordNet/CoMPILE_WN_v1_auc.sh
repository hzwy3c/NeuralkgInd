DATA_DIR=dataset

MODEL_NAME=CoMPILE
DATASET_NAME=WN18RR_v1
DATA_PATH=$DATA_DIR/$DATASET_NAME
DB_PATH=$DATA_DIR/WN18RR_v1_subgraph
TEST_DB_PATH=$DATA_DIR/WN18RR_v1_ind/test_subgraphs
PK_PATH=$DATA_DIR/WN18RR_v1.pkl
TRAIN_SAMPLER_CLASS=DglSampler2
VALID_SAMPLER_CLASS=ValidDglSampler2
TEST_SAMPLER_CLASS=NewDglSampler2
LITMODEL_NAME=indGNNLitModel
LOSS=Margin_Loss
MAX_EPOCHS=30
EMB_DIM=32
TRAIN_BS=16
EVAL_BS=16
TEST_BS=1
NUM_NEG=100
MARGIN=10.0
LR=1e-3 
CHECK_PER_EPOCH=1
EARLY_STOP_PATIENCE=20
NUM_WORKERS=20
DROPOUT=0
CALC_HITS=1,5,10
GPU=1
CHECKPOINT_DIR=/home/lli/NeuralKG-ind-NeuralKG-ind/output/link_prediction/WN18RR_v1/CoMPILE/epoch\=24-Eval\|aoc\=0.978.ckpt
TEST_METRIC=auc

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --db_path $DB_PATH \
    --pk_path $PK_PATH \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --valid_sampler_class $VALID_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --litmodel_name $LITMODEL_NAME \
    --loss $LOSS \
    --test_db_path $TEST_DB_PATH \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --test_bs $TEST_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LR \
    --check_per_epoch $CHECK_PER_EPOCH \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --num_workers $NUM_WORKERS \
    --dropout $DROPOUT \
    --calc_hits $CALC_HITS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --test_metric $TEST_METRIC \
    --test_only