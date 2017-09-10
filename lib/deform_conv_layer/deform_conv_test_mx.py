from __future__ import absolute_import
import os
import mxnet as mx
import numpy as np

gpu_device=mx.gpu()
cpu_device=mx.cpu()

# trans = np.random.rand(1,2,2,2)

if not os.path.isfile('test.npz'):
  with open("test.npz", 'wb') as f:
    arr=np.random.random((8, 6, 4, 5))
    np.save(f, arr)
else:
  with open("test.npz", 'rb') as f:
    arr = np.load(f)
kernel = mx.nd.array(np.ones((21,2,2,2)), ctx=gpu_device)
trans = mx.nd.array(np.ones((8,8,2,2)), ctx=gpu_device)
arr = mx.nd.array(arr, ctx=gpu_device)
data_grad = mx.nd.zeros_like(arr)
kernel_grad = mx.nd.zeros_like(kernel)
trans_grad = mx.nd.zeros_like(trans)

def main():
    data_var = mx.symbol.Variable('data')
    ker_var = mx.symbol.Variable('kernel')
    trans_var = mx.symbol.Variable('trans')
    res = mx.contrib.sym.DeformableConvolution(data=data_var, offset=trans_var, weight=ker_var, 
                                          num_group=3, no_bias=True, kernel=[2,2], num_filter=21, stride=[2, 2])
    rua = res.bind(ctx=gpu_device, args={'data':arr, 'kernel':kernel, 'trans':trans}, args_grad={'data':data_grad, 'kernel':kernel_grad, 'trans':trans_grad})
    rua.forward(is_train=True)
    rua.backward(out_grads=mx.nd.ones((8,21,2,2)))
    # print(trans.asnumpy())
    # res_arr = rua.outputs[0].asnumpy()
    # print(res_arr)
    # print([a.asnumpy() for a in rua.grad_arrays])
    print(data_grad.asnumpy())
    

if __name__ == '__main__':
    main()