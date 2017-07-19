from __future__ import absolute_import
import tensorflow as tf
from tensorflow.python.framework import ops
from . import deform_conv_op

@ops.RegisterGradient("DeformConvOp")
def _deform_conv_grad(op, grad):
  """The gradients for `deform_conv`.
  Args:
    op: The `deform_conv` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  filter = op.inputs[1]
  offset = op.inputs[2]
  
  strides = op.get_attr('strides')
  rates = op.get_attr('rates')
  num_groups = op.get_attr('num_groups')
  padding = op.get_attr('padding')
  data_format = op.get_attr('data_format')

  # compute gradient
  data_grad = deform_conv_op.deform_conv_grad_op(data, filter, offset, grad, strides, rates, num_groups, padding, data_format)

  return data_grad