import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg


class VGGnet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
        # n_classes = 21
        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
         .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
         .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3'))

        (self.feed('conv5_3')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales) * 3 * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score'))

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred'))

        #  shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer(len(anchor_scales) * 3 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))

        (self.feed('conv5_3')
            .conv(3, 3, 72, 1, 1, biased=True, rate=2, relu=False, name='conv6_1_offset', padding='SAME', initializer='zeros'))
        (self.feed('conv5_3', 'conv6_1_offset')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=True, num_deform_group=4, name='conv6_1'))
        (self.feed('conv6_1')
            .conv(3, 3, 72, 1, 1, biased=True, rate=2, relu=False, name='conv6_2_offset', padding='SAME', initializer='zeros'))
        (self.feed('conv6_1', 'conv6_2_offset')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=True, num_deform_group=4, name='conv6_2'))
        (self.feed('conv6_2', 'rois')
            .deform_psroi_pool(group_size=1, pooled_size=7, sample_per_part=4, no_trans=True, part_size=7, output_dim=256, trans_std=1e-1, spatial_scale=0.0625, name='offset_t')
            .fc(num_out=7 * 7 * 2, name='offset', relu=False)
            .reshape(shape=(-1,2,7,7), name='offset_reshape'))
        (self.feed('conv6_2', 'rois', 'offset_reshape')
            .deform_psroi_pool(group_size=1, pooled_size=7, sample_per_part=4, no_trans=False, part_size=7, output_dim=256, trans_std=1e-1, spatial_scale=0.0625, name='pool_6')
            .fc(4096, name='fc6')
            .dropout(0.5, name='drop6')
            .fc(4096, name='fc7')
            .dropout(0.5, name='drop7')
            .fc(n_classes, relu=False, name='cls_score')
            .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes*4, relu=False, name='bbox_pred'))

