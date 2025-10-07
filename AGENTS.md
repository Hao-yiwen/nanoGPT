# Repository Guidelines

## 项目结构与模块组织
主训练循环位于 `train.py`，Transformer 结构和相关工具集中在 `model.py`。预设配置放在 `config/`，数据准备脚本位于 `data/`（示例：`data/shakespeare_char/prepare.py`）。训练生成的检查点与采样结果请保存在各自的输出目录，例如 `out-shakespeare-char/`。图片、基准图表等素材保持在 `assets/`，若需更新笔记本 (`*.ipynb`)，请完整复现流程后再提交。

## 构建、测试与开发命令
推荐使用 `python -m venv .venv && source .venv/bin/activate` 创建隔离环境，然后依据 `README.md` 安装依赖。通过 `python data/shakespeare_char/prepare.py` 准备示例数据；使用 `python train.py config/train_shakespeare_char.py` 启动字符级实验；若要在多卡上训练 OpenWebText，可运行 `torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py`。完成训练后可用 `python sample.py --out_dir=out-shakespeare-char` 生成样本以快速校验。

## 代码风格与命名约定
遵循 PEP 8（4 空格缩进），变量命名优先与配置键保持一致（如 `n_layer`、`block_size`）。确保模块可被导入，避免在 import 阶段执行额外逻辑；辅助函数就近放置在使用位置附近。扩展配置时沿用现有命令行参数风格，坚持使用 snake_case，避免 camelCase。对非显而易见的实现添加简洁注释。

## 测试指引
仓库尚无正式测试套件，提交前至少运行一次轻量冒烟训练，例如 `python train.py config/train_shakespeare_char.py --max_iters=200`。确认采样脚本正常工作（`python sample.py --out_dir=<run-dir>`），并关注损失曲线是否异常。若修改涉及性能，建议在可用 GPU 上通过 `python bench.py --device=cuda` 复核吞吐。

## 提交与合并请求规范
延续现有的 conventional commit 风格（如 `feat: …`、`fix: …`、`docs: …`）。PR 描述需说明变更动机、关联 Issue，并突出 API 或配置差异。若影响训练表现，请附上修改前后的指标或输出。提交前再次验证受影响脚本或配置能通过文档中的命令顺利执行。
