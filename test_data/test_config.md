
# 200k数据集：--lr 3e-4 --batch-size 64

testA : --d-model 256 --layers 3  //train + valid_loss 1.5卡死
testB : --d-model 384 --layers 4  //train下降 ， 但valid_loss 1.4卡死
testC : --d-model 512 --layers 6  //valid_loss 飙升到10 

## 降低lr
testD : --d-model 512 --layers 6 --lr 1e-4  //train下降，但valid_loss 1.5卡死，泛化性缺失

