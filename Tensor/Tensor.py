import torch
import torch.tensor

'''
---------------------------------------------------------------------------
Data type               |   CPU tensor          |   GPU tensor             |
---------------------------------------------------------------------------
32-bit floating point	|   torch.FloatTensor   |   torch.cuda.FloatTensor |
---------------------------------------------------------------------------
64-bit floating point   |   torch.DoubleTensor	|   torch.cuda.DoubleTensor|
---------------------------------------------------------------------------
16-bit floating point	|   N/A	                |   torch.cuda.HalfTensor  |
---------------------------------------------------------------------------
8-bit integer (unsigned)|	torch.ByteTensor	|   torch.cuda.ByteTensor  |
---------------------------------------------------------------------------
8-bit integer (signed)	|   torch.CharTensor	|   torch.cuda.CharTensor  |
---------------------------------------------------------------------------
16-bit integer (signed)	|   torch.ShortTensor	|   torch.cuda.ShortTensor |
---------------------------------------------------------------------------
32-bit integer (signed)	|   torch.IntTensor	    |   torch.cuda.IntTensor   |
---------------------------------------------------------------------------
64-bit integer (signed)	|   torch.LongTensor	|   torch.cuda.LongTensor  |
---------------------------------------------------------------------------
'''

# 一个张量tensor可以从Python的list或序列构建：
a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])

print(a)
print(a.data)
print(a.shape)
print(a.dim())

# 一个空张量tensor可以通过规定其大小来构建：
b = torch.IntTensor(2, 4).zero_()
print(b)

# 可以用python的索引和切片来获取和修改一个张量tensor中的内容：
x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
print(x)
print(x[1][2])

# 每一个张量tensor都有一个相应的torch.Storage用来保存其数据。类tensor提供了一个存储的多维的、横向视图，并且定义了在数值运算。
# torch.FloatTensor.abs_()会在原地计算绝对值，并返回改变后的tensor，
# 而tensor.FloatTensor.abs()将会在一个新的tensor中计算结果。


# 根据可选择的大小和数据新建一个tensor。
# 如果没有提供参数，将会返回一个空的零维张量。
# 如果提供了numpy.ndarray,torch.Tensor或torch.Storage，
# 将会返回一个有同样参数的tensor.如果提供了python序列，将会从序列的副本创建一个tensor。

c = torch.tensor([[1], [2], [3]])
print(c.size())

c = c.expand(3, 4)
print(c)
print(c.size)