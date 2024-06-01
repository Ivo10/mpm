import struct
import torch


def face2list(cell_faces_num: str, cell_faces_indices: str) -> list:
    with open(cell_faces_num, 'rb') as file1:
        file1.seek(128)  # 跳过128字节的文件头
        values = []
        offset = 128
        while True:
            data = file1.read(4)
            if not data:
                break
            face_num = struct.unpack('I', data)[0]  # face_num是第n个cell包含的face个数
            # print(face_naum)

            with open(cell_faces_indices, 'rb') as file2:
                file2.seek(offset)
                value = set()
                for _ in range(face_num):
                    face_index = file2.read(4)
                    value.add(struct.unpack('I', face_index)[0])
                offset += face_num * 4  # 每次offset加上4个字节的int类型

            values.append(value)

    return values


def list2edge_index(cell_faces_num: str, cell_faces_indices: str) -> torch.Tensor:
    lst = face2list(cell_faces_num, cell_faces_indices)
    edge_index = []
    edge_index.append(torch.tensor([], dtype=torch.long))
    edge_index.append(torch.tensor([], dtype=torch.long))
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            # 如果两个cell的face集合交集不为空，则向edge_index中添加[i, j]和[j, i]
            if lst[i].intersection(lst[j]):
                edge_index[0] = torch.cat((edge_index[0], torch.tensor([i])))
                edge_index[0] = torch.cat((edge_index[0], torch.tensor([j])))
                edge_index[1] = torch.cat((edge_index[1], torch.tensor([j])))
                edge_index[1] = torch.cat((edge_index[1], torch.tensor([i])))
    edge_index = torch.stack(edge_index, dim=0)
    # print(edge_index_tensor)

    return edge_index


if __name__ == '__main__':
    print(list2edge_index('../datasets/input/geom/cell_faces_num.dat',
                          '../datasets/input/geom/cell_faces_indices.dat'))
