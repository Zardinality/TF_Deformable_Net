from __future__ import absolute_import
import tensorflow as tf
from tensorflow.python.framework import ops
import deform_psroi_pooling_op
import pdb


# @tf.RegisterShape("DeformPSROIPool")
# def _deform_psroi_pool_shape(op):
#   """Shape function for the DeformPSROIPool op.

#   """
#   dims_data = op.inputs[0].get_shape().as_list()
#   channels = dims_data[3]
#   dims_rois = op.inputs[1].get_shape().as_list()
#   num_rois = dims_rois[0]
#   output_dim = op.get_attr('output_dim')
#   group_size  = op.get_attr('group_size')
#   pooled_height = group_size
#   pooled_width = group_size

#   output_shape = tf.TensorShape([num_rois, pooled_height, pooled_width, output_dim])
#   return [output_shape, output_shape]

@ops.RegisterGradient("DeformPSROIPool")
def _deform_psroi_pool_grad(op, grad, _):
  """The gradients for `Deform_PSROI_pool`.
  Args:
    op: The `roi_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """

  
  data = op.inputs[0]
  rois = op.inputs[1]
  trans = op.inputs[2]  
  mapping_channel = op.outputs[1]
  spatial_scale = op.get_attr('spatial_scale')
  output_dim = op.get_attr('output_dim')
  group_size = op.get_attr('group_size')
  pooled_size = op.get_attr('pooled_size')
  part_size = op.get_attr('part_size')
  sample_per_part = op.get_attr('sample_per_part')
  trans_std = op.get_attr('trans_std')
  no_trans = op.get_attr('no_trans')
  
  

  # compute gradient
  #data_grad = psroi_pooling_op.psroi_pool_grad(data, rois, argmax, grad, pooled_height, pooled_width, spatial_scale)
  data_grad, trans_grad = deform_psroi_pooling_op.deform_psroi_pool_grad(data, rois, trans, mapping_channel, grad, spatial_scale, 
                                                            output_dim, group_size, pooled_size, part_size, sample_per_part,
                                                            trans_std, no_trans)
  # rois_grad = tf.zeros(rois.shape)
  return [data_grad, None, trans_grad]  # List of one Tensor, since we have one input

