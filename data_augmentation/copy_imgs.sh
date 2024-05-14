#! /bin/zsh

size=$(ls ../AirbusShipDetection/train_v2/ | wc -w | tr -d ' ')
let num=0

for file in $(ls ./train_v2/)
do
    cp ../AirbusShipDetection/train_v2/$file ./imgs
    num=$num+1
    if [ $num -ge 20 ]
    then
        exit
    fi
done