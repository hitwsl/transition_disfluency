#!/bin/bash
python ./python/lstm-ner.py -T ./data/eng.train \
    -d ./data/eng.dev \
    -w ./data/sskip.100.vectors \
    --hidden_dim 100 \
    --lstm_input_dim 100 \
    --pretrained_dim 100 \
    --action_dim 20 \
    --maxiter 10 \
    --graph chris \
    -t -P --conlleval "./conlleval"
