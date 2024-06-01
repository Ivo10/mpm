import struct


def gen_res_dat(y, file_path):
    # 确保tensor是一个1D的浮点数tensor
    tensor = y.float().flatten()

    # 创建文件头
    header = f'offset=128\nunits=\nncols=1\ndim={tensor.shape[0]}\ntype=float\nnbytes=4\nbig=0\n'
    header = header.ljust(128, '\x00')  # 将头部信息填充至128字节

    # 将tensor转换为numpy数组
    numpy_array = tensor.numpy()

    # 使用小端格式将numpy数组转换为二进制数据
    binary_data = struct.pack(f'<{len(numpy_array)}f', *numpy_array)

    # 写入文件
    with open(file_path, 'wb') as file:
        file.write(header.encode('utf-8'))  # 写入头部信息
        file.write(binary_data)  # 写入二进制数据
