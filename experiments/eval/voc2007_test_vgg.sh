 python ./faster_rcnn/test_net.py \
      --gpu 0 \
       --weights ./output/faster_rcnn_voc_vgg/voc_2007_trainval \
        --imdb voc_2007_test \
         --cfg ./experiments/cfgs/faster_rcnn_end2end_resnet.yml \
          --network VGGnet_test
