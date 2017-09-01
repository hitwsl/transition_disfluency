#!/bin/bash
cat train | python trans.py > trains.disf
cat develop | python trans.py > devs.disf
cat test | python trans.py > tests.disf
cat trains.disf | python get_feature_all.py > features_train_ord_disf
cat devs.disf | python get_feature_all.py > features_dev_ord_disf
cat tests.disf | python get_feature_all.py > features_test_ord_disf
rm trains.disf devs.disf tests.disf
cat features_dev_ord_disf | python convert-conll2trans.py > dev.trans.disf
python conll2parser.py -f dev.trans.disf > dev.ord.parser.disf
rm dev.trans.disf
cat features_test_ord_disf | python convert-conll2trans.py > dev.trans.disf
python conll2parser.py -f dev.trans.disf > test.ord.parser.disf
rm dev.trans.disf
cat features_train_ord_disf | python convert-conll2trans.py > dev.trans.disf
python conll2parser.py -f dev.trans.disf > train.ord.parser.disf
rm dev.trans.disf features* 
