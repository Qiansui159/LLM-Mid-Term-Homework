# Transformer-Seq2Seq: 中英文机器翻译

## 项目简介

本项目实现了基于 Transformer 架构的中英文机器翻译模型，包含完整训练、推理、可视化流程。代码结构清晰，易于复现和扩展。

- 代码目录：`src/`
- 依赖文件：`requirements.txt`
- 运行脚本：`scripts/train.sh`
- 结果目录：`results/`（保存训练曲线和表格）

## 硬件要求

- 推荐 GPU：NVIDIA RTX 3060 及以上，显存 8GB+
- 支持 CPU 运行（速度较慢）
- CUDA 11.8 及以上（如用 GPU）

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据准备

请将预处理好的中英文数据集（如 cmn.txt）放在 `data/` 目录下，或修改命令行参数指定路径。

数据集下载链接：[link](http://www.manythings.org/anki/)

## 训练命令（可复现实验，含随机种子）

```bash
# 推荐 exact 命令（可复现论文结果）
python train.py \
	--data_dir ./data/cmn.txt \
	--results_dir ../results/run_001 \
	--batch_size 64 \
	--num_epochs 15 \
	--d_model 512 \
	--num_heads 8 \
	--d_ff 2048 \
	--num_layers 6 \
	--dropout 0.1 \
	--learning_rate 1e-4 \
	--pos_state True \
	--seed 42
```

或直接运行脚本：

```bash
bash train.sh
```

## 结果查看

- 训练日志与模型参数保存在 `results/run_XXX/` 目录
- 损失曲线图：`results/run_XXX/loss_curve.png`
- 最优模型参数：`results/run_XXX/best_model.pth`

## 复现实验说明

- 所有实验均设置随机种子（`--seed 42`），保证结果可复现
- 训练曲线和表格自动保存到 `results/` 目录
- 支持多次实验，自动编号 run_001、run_002 等

## 参考命令

- 查看所有参数：
	```bash
	python src/train.py --help
	```

## 主要依赖

- torch==2.0.0+cu118
- torchtext==0.15.1
- numpy, pandas, matplotlib, tqdm, nltk 等

## 结果示例

![loss curve](./results/run_001/loss_curve.png)

---

如需更多细节，请查阅代码注释和 `src/` 目录下各模块。
