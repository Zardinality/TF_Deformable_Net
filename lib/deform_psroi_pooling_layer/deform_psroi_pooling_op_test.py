from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import deform_psroi_pooling_op
import deform_psroi_pooling_op_grad
import pdb

# pdb.set_trace()
data_arr = np.random.rand(1,25,5,5)
# roi = np.array([[0, 0, 0, 4, 4]],dtype=np.float32)
trans_arr = np.random.rand(1,2,2,2)

# with open("data.npz", 'rb') as f:
#     data_arr = np.load(f)
# with open("trans.npz", 'rb') as f:
#     trans_arr = np.load(f)


rois = tf.convert_to_tensor([ [0, 0, 0, 4, 4]], dtype=tf.float32)
trans = tf.convert_to_tensor(trans_arr, dtype=tf.float32)
hh=tf.convert_to_tensor(data_arr,dtype=tf.float32)
[y2, channels] = deform_psroi_pooling_op.deform_psroi_pool(hh, rois, trans=trans, pooled_size=2, output_dim=1, group_size=1, spatial_scale=1.0, 
                                                           trans_std=1e-1, sample_per_part=1, part_size=2, no_trans=False)
s = tf.gradients(y2, [hh, trans])
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess.run(s[0])
# print( sess.run(trans))
# print( sess.run(y2))
print( sess.run(s[1]))
# print( sess.run(s[1]))
# pdb.set_trace()
