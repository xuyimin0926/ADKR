import torch

for i in range(14):
    load_torch = torch.load('test_save_tensor_{}.pt'.format(i)) # 读取

    print(load_torch.shape)



    # 统计非零元素的个数和总元素个数
    non_zero_count = 0
    total_count = 0

    # 遍历张量并统计非零元素
    for dim1 in load_torch:
        for dim2 in dim1:
            for dim3 in dim2:
                for element in dim3:
                    if element != 0:
                        non_zero_count += 1
                    total_count += 1

    # 计算非零元素的比例
    non_zero_ratio = non_zero_count / total_count

    print("非零元素的比例:", non_zero_ratio)
    #torch.set_printoptions(profile="full")
    #print(load_torch.shape)
