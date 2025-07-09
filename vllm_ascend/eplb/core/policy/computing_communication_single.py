
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger("msit_logger")

def save_matrix_to_json(output_path, file_name, deployment):
    # 构建两层嵌套字典
    num_layers = deployment.shape[0]
    num_cards = deployment.shape[1]

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            # 将 1*4 的行矩阵转换为列表
            device = {"device_id": j, "device_expert": deployment[i, j].tolist()}
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = f"{output_path}{file_name}.json"

    # 保存为 JSON 文件
    try:
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"xie文件 {deployment} 时出错: {e}")

def compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert, is_only):
    route_expert_num = len(origin_weights)
    route_expert_redundancy = [[] for _ in range(route_expert_num)]
    for i in range(num_redundancy_expert):
        sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
        weights = [origin_weights[idx] for idx in sorted_indices]
        tmp_raw_weight = weights[0][1] * (len(route_expert_redundancy[weights[0][0]]) + 1)
        route_expert_redundancy[weights[0][0]].append(route_expert_num + i)
        avg_weight = tmp_raw_weight / (len(route_expert_redundancy[weights[0][0]]) + 1)
        weights[0] = (weights[0][0], avg_weight)
        origin_weights = weights

    expert_num = route_expert_num + num_redundancy_expert
    if card_num == 0:
        raise RuntimeError("card_num can not be 0.")
    items_per_box = expert_num // card_num
    remaining_items = expert_num % card_num

    boxes = [[] for _ in range(card_num)]
    boxes_weights = [[] for _ in range(card_num)]
    box_weights = [0] * card_num
    box_counts = [0] * card_num

    all_weights = np.zeros((expert_num,), dtype='object')
    all_weights[: route_expert_num] = origin_weights

    index = route_expert_num
    for i in range(route_expert_num):
        redundancy_num = len(route_expert_redundancy[i])
        for _ in range(redundancy_num):
            for item, weight in origin_weights:
                if item == i:
                    all_weights[index] = (item, weight)
                    index += 1

    sorted_indices = np.argsort([t[1] for t in all_weights], kind='stable')[::-1]
    all_weights = [all_weights[idx] for idx in sorted_indices]
    for item_id, weight in all_weights:
        min_box_index = -1
        for i in range(card_num):
            if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                if min_box_index == -1 or box_weights[i] < box_weights[min_box_index]:
                    if item_id not in boxes[i]:
                        min_box_index = i

        boxes[min_box_index].append(item_id)
        boxes_weights[min_box_index].append(weight)
        box_weights[min_box_index] += weight
        box_counts[min_box_index] += 1

        if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
            remaining_items -= 1

    result = []
    for i in range(card_num):
        result.append({
            "box_index": i + 1,
            "items": boxes[i],
            "weight": boxes_weights[i],
            "total_weight": box_weights[i],
            "item_count": box_counts[i]
        })

    return result, boxes


# 冗余专家部署
def lb_and_intra_layer_affinity_redundancy_deploy(
        layer_workloads,  
        num_redundancy_expert, 
        num_npus=64, 
        num_original_expert=256,):
    """
    :param layer_workloads[layer_num, expert_num] 58*256
    :return: optimized layer_deployment: [layer_num, card_num, card_expert_num] 58*64*4
    """
    layer_num = layer_workloads.shape[0]
    expert_num = layer_workloads.shape[1]
    if num_original_expert != expert_num:
        raise ValueError(f"原始专家数量 {num_original_expert} 必须等于 expert_num {expert_num}")

    if num_npus <= 0:
        raise ValueError("NPUs 数量必须大于 0")

    if num_npus < num_redundancy_expert:
        raise ValueError(f"NPUs 数量 {num_npus} 必须大于或等于冗余专家数量 {num_redundancy_expert}")

    global_deployment = [[[] for _ in range(num_npus)] for _ in range(layer_num)]

    original_weights = []
    max_weights = []
    average_weights = []
    y_list = []
    for layer in range(layer_num):
        weights = np.zeros((expert_num,), dtype='object')
        for expert_id, workload_weight in enumerate(layer_workloads[layer]):
            weights[expert_id] = (expert_id, workload_weight)

        result, layer_deployment = compute_balanced_pack_redundancy(weights, num_npus, num_redundancy_expert, 0)

        max_weight = 0
        for box in result:
            if max_weight < box['total_weight']:
                max_weight = box['total_weight']
            #print(layer,
            #    f"before: Box {box['box_index']}: "
            #    f"Items = {box['items']}, weight = {box['weight']}, "
            #    f"Total Weight = {box['total_weight']}, Item Count = {box['item_count']}"
            #)

        new_value = layer_workloads[layer].reshape(num_npus, -1)
        ave_workload = np.sum(layer_workloads[layer]) / num_npus
        row_sum = np.sum(new_value, axis=1)
        original_weights.append(row_sum.max() / ave_workload)
        max_weights.append(max_weight / ave_workload)
        average_weights.append(1)
        print("imbalance: ", layer, row_sum.max()/ave_workload, max_weight/ave_workload)

        global_deployment[layer] = layer_deployment

    y_list.append(original_weights)
    y_list.append(max_weights)
    y_list.append(average_weights)

    return global_deployment, y_list

def calculate_average(lst):
    """计算一维列表的平均值"""
    if not lst:
        raise ValueError("列表不能为空")

    total = 0
    count = 0

    for element in lst:
        # 检查元素是否为数值类型
        if isinstance(element, (int, float, np.int64, np.int32, np.float64)):
            total += element
            count += 1
        else:
            # 非数值类型元素会被忽略，并打印警告
            print(f"警告: 元素 {element} 不是数值类型，已被忽略")

    if count == 0:
        raise ValueError("列表中不包含任何数值类型的元素")

    return total / count

def layer_imblance_polt(y_list, label_names, device_num, output_path, file_name):

    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    x = [i for i in range(58)]
    for index, y in enumerate(y_list):
        plt.plot(x, y, label=rf'{label_names[index]}, avg={calculate_average(y)}')

    # 显示图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title(rf'imbalance (device_num={device_num})')
    plt.xlabel('layer')
    plt.ylabel(f'imbalance')

    # 显示网格线
    plt.grid(True)

    plt.savefig(output_path + file_name, dpi=300)

    # 清理当前图表
    plt.close()

if __name__ == '__main__':
    import sys
    temp = "online_data"
    input_path = f"D:/code2025/dynamic_c2lb/eplb-static-ep-gather-heart-message/merged_data/merged_data_decode.pt"
    output_path = f"D:/code2025/dynamic_c2lb/eplb-static-ep-gather-heart-message/merged_data/"
    num_redundancy_expert = 32
    num_npus = 32
    workloads = np.zeros((1, 58, 256))

    path_file = input_path
    workloads = torch.load(path_file, map_location=torch.device('cpu')).float().int().numpy()
    global_deployment, y_list = lb_and_intra_layer_affinity_redundancy_deploy(workloads, num_redundancy_expert, num_npus,256)
    file_name = f"delta_{temp}_{num_npus}_{num_redundancy_expert}"
    save_matrix_to_json(output_path, file_name, np.array(global_deployment))
    label_names = [f'baseline imbalance',
                    f'eplb imbalance',
                    f'average imbalance']
    new_file_name = f"delta_{temp}_{num_npus}_{num_redundancy_expert}.png"
    layer_imblance_polt(y_list, label_names, num_npus, output_path, new_file_name)
