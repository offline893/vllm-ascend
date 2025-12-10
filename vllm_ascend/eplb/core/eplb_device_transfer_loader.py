#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from enum import Enum

import torch.distributed as dist
from vllm.logger import logger


class ExpertWeightUpdateState(Enum):
    WAITING = 0  # waiting for updated expert_map by EplbWorker
    READY = 1  # ready for d2d expert weights updating
    TRANSFERRING = 2  # d2d finished and waiting for updating expert_map into model


class D2DExpertWeightLoader:

    def __init__(self):
        self.comm_op_list = None
        self.updated_expert_map = None
        self.updated_log2phy_map = None
        self.layer_id = -1  # layer id to be updated
        self.state = ExpertWeightUpdateState.WAITING
        self.recv_expert_list = []
        self.mock_flag = True

    def set_adator(self, eplb_adaptor):
        self.eplb_adaptor = eplb_adaptor

    # from vllm.utils.profiling import cprofile
    # @cprofile("generate_expert_d2d_transfer_task.prof")
    # def generate_expert_d2d_transfer_task(self, expert_send_info,
    #                                       expert_recv_info, updated_expert_map,
    #                                       layer_id):
    #     # When current send/recv and weight.expert_map update tasks are not finished, cannot accept new d2d task
    #     if self.state != ExpertWeightUpdateState.WAITING:
    #         logger.warning_once(
    #             "current d2d weight update tasks are on-going, cannot accept new weight update task"
    #         )
    #         return
    #
    #     # If neither send nor receive task is needed for this layer on this rank, return
    #     if not (expert_send_info or expert_recv_info):
    #         return
    #
    #     self.updated_expert_map = updated_expert_map
    #
    #     self.layer_id = layer_id
    #     self.comm_op_list = []
    #     for send_info in expert_send_info:
    #         dst_rank, global_expert_id_to_send = send_info
    #         local_expert_id = self.eplb_adaptor.expert_map_per_layer_cpu[
    #             layer_id][global_expert_id_to_send].item()
    #         for src_tensor in self.eplb_adaptor.expert_param_per_layer[
    #                 layer_id][local_expert_id]:
    #             src_tensor = src_tensor.clone()
    #             self.comm_op_list.append(
    #                 dist.P2POp(dist.isend, src_tensor, dst_rank))
    #
    #     buffer_tensor_id = 0
    #     for recv_info in expert_recv_info:
    #         recv_rank, global_expert_id_to_recv = recv_info
    #         for buffer_tensor in self.eplb_adaptor.buffer_tensor_list[
    #                 buffer_tensor_id]:
    #             self.comm_op_list.append(
    #                 dist.P2POp(dist.irecv, buffer_tensor, recv_rank))
    #         local_expert_to_replace = self.updated_expert_map[
    #             global_expert_id_to_recv].item()
    #         self.recv_expert_list.append(
    #             (local_expert_to_replace, buffer_tensor_id))
    #         buffer_tensor_id += 1
    #
    #     self.state = ExpertWeightUpdateState.READY

    from vllm.utils.profiling import cprofile
    @cprofile("generate_expert_d2d_transfer_task.prof")
    def generate_expert_d2d_transfer_task(self, expert_send_info,
                                          expert_recv_info, updated_expert_map,
                                          layer_id):
        # 1. 快速状态检查 (保持不变)
        if self.state != ExpertWeightUpdateState.WAITING:
            logger.warning_once(
                "current d2d weight update tasks are on-going, cannot accept new weight update task"
            )
            return

        if not (expert_send_info or expert_recv_info):
            return

        self.updated_expert_map = updated_expert_map
        self.layer_id = layer_id

        # 2. 预取频繁访问的数据结构，减少循环内的属性查找开销
        # 假设 expert_map_per_layer_cpu 确实在 CPU 上，如果不在，建议在初始化时就 .cpu()
        layer_expert_map = self.eplb_adaptor.expert_map_per_layer_cpu[layer_id]
        layer_params = self.eplb_adaptor.expert_param_per_layer[layer_id]
        buffer_list = self.eplb_adaptor.buffer_tensor_list

        # 预分配列表空间（微小优化，Python list append 已经很快了，但这样更整洁）
        self.comm_op_list = []

        # --- 发送逻辑优化 ---
        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info

            # 优化点：直接获取 int，避免潜在的 tensor 操作
            # 确保 layer_expert_map 是 CPU tensor 或 numpy 数组
            local_expert_id = layer_expert_map[global_expert_id_to_send].item()

            # 获取该专家对应的参数列表
            expert_params = layer_params[local_expert_id]

            for src_tensor in expert_params:
                # 🔴 关键优化：移除 .clone()
                # 直接发送原始 tensor。dist.isend 会增加 tensor 的引用计数，
                # 只要在通信完成前不执行 in-place update (如 optimizer.step)，就是安全的。
                self.comm_op_list.append(
                    dist.P2POp(dist.isend, src_tensor, dst_rank))

        # --- 接收逻辑优化 ---
        buffer_tensor_id = 0
        for recv_info in expert_recv_info:
            recv_rank, global_expert_id_to_recv = recv_info

            # 缓存当前 buffer 列表
            current_buffers = buffer_list[buffer_tensor_id]

            for buffer_tensor in current_buffers:
                self.comm_op_list.append(
                    dist.P2POp(dist.irecv, buffer_tensor, recv_rank))

            local_expert_to_replace = self.updated_expert_map[
                global_expert_id_to_recv].item()

            self.recv_expert_list.append(
                (local_expert_to_replace, buffer_tensor_id))
            buffer_tensor_id += 1

        self.state = ExpertWeightUpdateState.READY

    def set_log2phy_map(self, log2phy_map):
        self.updated_log2phy_map = log2phy_map

    from vllm.utils.profiling import cprofile
    @cprofile("asyn_expert_weight_transfer.prof")
    def asyn_expert_weight_transfer(self, reqs):
        # Only when send/recv tasks are parsed into self.comm_op_list, d2d send/recv tasks can be luanched
        if self.state != ExpertWeightUpdateState.READY:
            return

        # set asynchronous stream for d2d expert weight transfer
        if self.comm_op_list:
            ret_list = dist.batch_isend_irecv(self.comm_op_list)
            reqs.extend(ret_list)

        self.state = ExpertWeightUpdateState.TRANSFERRING

    from vllm.utils.profiling import cprofile
    @cprofile("update_expert_map_and_weight.prof")
    def update_expert_map_and_weight(self, reqs):
        # Only after send/recv tasks have been luanched, expert_map and weight can be updated
        if self.state != ExpertWeightUpdateState.TRANSFERRING:
            return

        # Waiting for send/recv tasks finish
        for req in reqs:
            req.wait()

        if self.comm_op_list is not None:
            self.comm_op_list = None

        # update expert_map
        self.eplb_adaptor.do_update_expert_map(self.layer_id,
                                               self.updated_expert_map)

        # update log2phy_map
        self.eplb_adaptor.do_update_log2phy_map(self.layer_id,
                                                self.updated_log2phy_map)

        # update expert weight
        buffer_tensor_id = 0
        for recv_expert_info in self.recv_expert_list:
            local_expert_to_replace, buffer_tensor_id = recv_expert_info
            self.eplb_adaptor.do_update_expert_weight(self.layer_id,
                                                      local_expert_to_replace,
                                                      buffer_tensor_id)

        logger.info(
            f"[EPLB] finished update expert weight for layer: {self.layer_id}")

        self.recv_expert_list = []
        self.updated_expert_map = None
        self.layer_id = -1
        self.state = ExpertWeightUpdateState.WAITING

    def load_impl(self, old_expert_table, new_expert_table):
        raise NotImplementedError
