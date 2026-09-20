
# 200k数据集：--lr 3e-4 --batch-size 64

testA : --d-model 256 --layers 3  //train + valid_loss 1.5震荡
testB : --d-model 384 --layers 4  //train下降 ， 但valid_loss 1.4震荡
testC : --d-model 512 --layers 6  //valid_loss 飙升到10 

## 降低lr
testD : --d-model 512 --layers 6 --lr 1e-4  //train下降至1.1，但valid_loss 1.5震荡，泛化性缺失

## 数据增加至600k： 

testE : --d-model 512 --layers 6 --lr 1e-4  // loss到2左右开始下降缓慢，怀疑学习率不够。

## 600k ， 更改代码加入lr_warm，前10%增加，后续衰减到0 :

testF : --d-model 512 --layers 6 --lr 3e-4 --lr-warmup --warmup-ratio 0.1 //valid loss 稳定下降到约 1.39开始震荡，疑似衰减严重，导致无法训练。