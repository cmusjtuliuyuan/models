#!/bin/sh

python nets/benchmark.py --model_type inception_v1 --batch_size 128

python nets/benchmark.py --model_type inception_v2 --batch_size 128

python nets/benchmark.py --model_type inception_v3 --batch_size 128

python nets/benchmark.py --model_type alexnet_v2 --batch_size 256

python nets/benchmark.py --model_type resnet_v1_50 --batch_size 64

python nets/benchmark.py --model_type resnet_v1_101 --batch_size 64

python nets/benchmark.py --model_type resnet_v1_152 --batch_size 64

python nets/benchmark.py --model_type resnet_v1_200 --batch_size 64

python nets/benchmark.py --model_type resnet_v2_50 --batch_size 64

python nets/benchmark.py --model_type resnet_v2_101 --batch_size 64

python nets/benchmark.py --model_type resnet_v2_152 --batch_size 64

python nets/benchmark.py --model_type resnet_v2_200 --batch_size 64

python nets/benchmark.py --model_type vgg_16 --batch_size 64

python nets/benchmark.py --model_type vgg_19 --batch_size 64
