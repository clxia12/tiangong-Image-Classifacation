#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=/home/clxia/caffe-master/models/ATAnet/solver.prototxt \
    --weights=/home/clxia/caffe-master/models/ATAnet/ATAnet.caffemodel \
    $@
