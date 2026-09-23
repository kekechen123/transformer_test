
# 200k数据集：--lr 3e-4 --batch-size 64

testA : --d-model 256 --layers 3  //train + valid_loss 1.5震荡

testB : --d-model 384 --layers 4  //train下降 ， 但valid_loss 1.4震荡

testC : --d-model 512 --layers 6  //valid_loss 飙升到10 

## 降低lr
testD : --d-model 512 --layers 6 --lr 1e-4  //train下降至1.1，但valid_loss 1.5震荡。个人猜测：泛化性缺失

## 数据增加至600k： 

testE : --d-model 512 --layers 6 --lr 1e-4  // loss到2左右开始下降缓慢，怀疑学习率不够。

## 600k ，30epoch，学习率回复3e-4，但为了稳定加入lr_warmup，前10%增加，后续cos衰减到0 :

testF : --d-model 512 --layers 6 --lr 3e-4 --lr-warmup --warmup-ratio 0.1 //valid loss 稳定下降到约 1.39开始震荡。个人判断：疑似衰减严重，导致无法训练？

## 600k ，30epoch，保持lr_warmup 不变，但cos衰减删除，峰值后保持lr 3e-4

testG : --d-model 512 --layers 6 --lr 3e-4 --lr-warmup --warmup-ratio 0.1  //valid loss 下降明显比F慢，1.5卡死。 

## 惊人发现：新增的400k数据集中存在大量垃圾乱码！被坑了！！！重搞数据！！！用干净的数据复刻F测试，还是选择30个epoch。

testF+ : --d-model 512 --layers 6 --lr 3e-4 --lr-warmup --warmup-ratio 0.1 //效果十分显著，收敛速度变快且在15个epoch达到了1.34这个最好成绩，但下降也变得开始缓慢，我给终止了。

test_f+_clean： --d-model 512 --layers 6 --lr 3e-4 --lr-warmup --warmup-ratio 0.1 // 重复上个实验跑到了28epoch，停止在了1.26，感觉长句还不是很理想。
