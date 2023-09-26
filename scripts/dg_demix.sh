#!/bin/bash

cd ..

DATA=~/Documents/dataset
DASSL=~/Documents/CVPR2023/Dassl.pytorch-master

DATASET=pacs
TRAINER=Demix
NET=resnet18

PAR_DESC=0.2
LAM=0.5

if [ ${DATASET} == pacs ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == office_home_dg ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
fi


for SEED in $(seq 1 10)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
            LR=0.002
            BS=32
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
            LR=0.02
            BS=128
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
            LR=0.002
            BS=64
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
            LR=0.002
            BS=16
        fi
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/pacs.yaml \
        --config-file configs/trainers/demix/${DATASET}_random.yaml \
        --output-dir output/${DATASET}/${TRAINER}/${NET}/random/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET} \
        OPTIM.LR ${LR} \
        DATALOADER.TRAIN_X.BATCH_SIZE ${BS} \
        TRAINER.DEMIX.PAR_DESC ${PAR_DESC} \
        TRAINER.DEMIX.LAM ${LAM}
    done
done
