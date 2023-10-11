# LLM-RLHF-Tuning

本项目从零实现了RLHF三阶段训练，并在文档中详细写了实现细节，欢迎大家交流讨论[WeChat](assets/RLHF讨论群.png)

### 主要内容：
- 支持指令微调Alpaca模型
- 支持训练Reward模型
- 支持PPO算法训练RL模型
    - 支持基于两个基模型，两个lora的适配器，同时加载RM、SFT、Actor、Critic四个模型，支持accelerate分布式训练 （[PPO算法实现细节](https://zhuanlan.zhihu.com/p/649665766)）
    - 支持基于一个基模型，两个lora适配器，同时加载RM、SFT、Actor、Critic四个模型，支持accelerate、deepspeed训练
    - 支持基于一个基模型，一个lora适配器，Actor、Critic共享base model，同时实现RM、SFT、Actor、Critic四个模型功能，支持accelerate、deepspeed训练
- 支持DPO算法训练模型

### 更新
- [23/8/23] 支持LLaMA2模型训练；支持DPO训练；支持基于一个基模型、选择一个或两个lora适配器训练PPO、支持accelerate、deepspeed训练
- [23/8/13] 支持LLaMA模型训练；支持基于两个基模型、两个lora的适配器训练PPO；支持accelerate分布式训练


### 功能
与开源的RLHF训练框架的功能进行对比
| 框架               |      SFT Train     |       RM Train     |       PPO Train    |       DPO Train   |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| Our                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | 
| [Deepspeed-chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |
| [trl](https://github.com/huggingface/trl)            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)      |                    |                    | :white_check_mark: |                    |


##### PPO Train 
| 框架               |     Accelerate     |    Deepspeed       |     Multi LORA     |     最低模型参数量 (7B为例) |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | 
| Our                | :white_check_mark: | :white_check_mark: | :white_check_mark: | single model size ～ 7B | 
| [Deepspeed-chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) |                    | :white_check_mark: |                    | sft+rm+actor+critic ～ 28B |
| [trl](https://github.com/huggingface/trl)            | :white_check_mark: |            |             | single model size（not use ref model）～ 7B |
| [MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)      | actor model、critic model | sft model、rm model |                    | sft+rm+actor+critic ～ 28B |



## 使用指引

#### 环境搭建
```
accelerate==0.21.0
datasets==2.13.1
scikit-learn==1.3.0
sentencepiece==0.1.99
tqdm==4.65.0
transformers==4.31.0
wandb==0.15.8
peft==0.4.0
torch==2.0.1
trl==0.5.0
deepspeed==0.10.0
```

#### 支持模型
- LLaMA
- LLaMA2

#### 支持训练方式
- LoRA

## 训练细节
#### 指令微调模型
- [训练指南](https://github.com/Joyce94/LLM-RLHF-Tuning/wiki/%E6%8C%87%E4%BB%A4%E5%BE%AE%E8%B0%83%E6%A8%A1%E5%9E%8B)


#### 训练奖励模型
- [训练指南](https://github.com/Joyce94/LLM-RLHF-Tuning/wiki/%E8%AE%AD%E7%BB%83%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B)

#### PPO训练
- 训练指南
    - [基于两个基模型](https://github.com/Joyce94/LLM-RLHF-Tuning/wiki/PPO%E8%AE%AD%E7%BB%83%E2%80%90%E5%9F%BA%E4%BA%8E%E4%B8%A4%E4%B8%AA%E5%9F%BA%E6%A8%A1%E5%9E%8B)
        - [PPO算法实现细节](https://zhuanlan.zhihu.com/p/649665766)

    - [基于一个基模型](https://github.com/Joyce94/LLM-RLHF-Tuning/wiki/PPO%E8%AE%AD%E7%BB%83%E2%80%90%E5%9F%BA%E4%BA%8E%E4%B8%80%E4%B8%AA%E5%9F%BA%E6%A8%A1%E5%9E%8B)

#### DPO训练
- [训练指南](https://github.com/Joyce94/LLM-RLHF-Tuning/wiki/DPO%E8%AE%AD%E7%BB%83)

## TODO
- [x] 支持LLaMA2模型
- [x] 支持deepspeed训练
- [x] 支持DPO训练
- [ ] PPO提升训练稳定性，实现ppo-max
- [ ] 支持DDPO训练
- [ ] 支持[RRHF](https://github.com/GanjinZero/RRHF)
- [ ] 支持[RAFT](https://github.com/OptimalScale/LMFlow)
- [ ] 支持拒绝采样 RFT
- [ ] 支持BLOOM模型
- [ ] 支持Baichuan模型
- [ ] 支持QLoRA训练


欢迎加群讨论 [WeChat](assets/RLHF讨论群.png)




