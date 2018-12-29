#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/tianyi
DATA=data/tianyi
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/tianyi_train_lmdb \
  $DATA/ialt_MWI_imagenet_mean.binaryproto

echo "Done."
