from __future__ import absolute_import
import mxnet as mx
import numpy as np

gpu_device=mx.gpu()
# data = np.random.rand(1,25,5,5)
roi = mx.nd.array(np.array([[0, 0, 0, 4, 4]],dtype=np.float32), ctx=gpu_device)
# trans = np.random.rand(1,2,2,2)

with open("data.npz", 'rb') as f:
    data = mx.nd.array(np.load(f), ctx=gpu_device)
with open("trans.npz", 'rb') as f:
    trans = mx.nd.array(np.load(f), ctx=gpu_device)

data_grad = mx.nd.zeros_like(data)
roi_grad = mx.nd.zeros_like(roi)
trans_grad = mx.nd.zeros_like(trans)

def main():
    data_var = mx.symbol.Variable('data')
    roi_var = mx.symbol.Variable('roi')
    trans_var = mx.symbol.Variable('trans')
    res = mx.contrib.sym.DeformablePSROIPooling(data=data_var, rois=roi_var, trans=trans_var, group_size=1, pooled_size=2, 
                                          output_dim=1, no_trans=False, part_size=2, sample_per_part=1, spatial_scale=1., trans_std=0.1)
    rua = res.bind(ctx=gpu_device, args={'data':data, 'roi':roi, 'trans':trans}, args_grad={'data':data_grad, 'roi':roi_grad, 'trans':trans_grad})
    rua.forward(is_train=True)
    rua.backward(out_grads=mx.nd.ones((1, 1, 2, 2)))
    # print(trans.asnumpy())
    # res_arr = rua.outputs[0].asnumpy()
    # print(res_arr)
    # print([a.asnumpy() for a in rua.grad_arrays])
    print(trans_grad.asnumpy())
    

if __name__ == '__main__':
    main()