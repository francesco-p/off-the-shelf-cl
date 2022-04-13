#/bin/sh

for dataset in 'CUB200'
do
    for model in 'resnet18' 'resnet34' 'resnet50' 'resnet152' 'vit_tiny_patch16_224' 'vit_base_patch16_224' 'deit_tiny_patch16_224' 'deit_base_patch16_224' 
    do

        python off_the_shelf.py --model ${model} --dataset ${dataset}

    done
done
#'Core50' 'CUB200' 'OxfordFlowers102' 'TinyImageNet200' 
