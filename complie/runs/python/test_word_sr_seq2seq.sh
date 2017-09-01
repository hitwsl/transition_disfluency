#!/bin/bash
MODEL=$1

python ./python/lstm-ner.py -T ./data/eng.train \
    -d ./data/eng.dev \
    -w ./data/sskip.100.vectors \
    --hidden_dim 100 \
    --lstm_input_dim 100 \
    --pretrained_dim 100 \
    --action_dim 20 \
    --graph seq2seq \
    -m $MODEL \
    -P --conlleval "./conlleval"

python ./python/lstm-ner.py -T ./data/eng.train \
    -d ./data/eng.test \
    -w ./data/sskip.100.vectors \
    --hidden_dim 100 \
    --lstm_input_dim 100 \
    --pretrained_dim 100 \
    --action_dim 20 \
    --graph seq2seq \
    -m $MODEL \
    -P --conlleval "./conlleval"
