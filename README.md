# MTVD

### 基线模型复现

若要复现基线模型，我们推荐使用作者发布的原始版本或者 CodeXGLUE 仓库。 这里我们使用 CodeBERT 为例来介绍微调和推理脚本的使用。

**微调**
```bash
cd CodeBERT\code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=dataset/BigVul/train.jsonl \
    --eval_data_file=dataset/BigVul/valid.jsonl \
    --test_data_file=dataset/BigVul/test.jsonl \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```

**推理**
```bash
cd CodeBERT\code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_eval \
    --do_test \
    --train_data_file=dataset/BigVul/train.jsonl \
    --eval_data_file=dataset/BigVul/valid.jsonl \
    --test_data_file=dataset/BigVul/test.jsonl \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

### 数据处理

`graph_process` 模块的代码提供了使用 JOERN 处理函数源码得到代码属性图的实现。运行的脚本为 `/graph_process/graph_process.py`，
```bash
python graph_process.py  --xxx --xxx (**kwargs)
```
其中可以设置的参数如下
```python
dataset='bigvul',  # 数据集名称
group='group0',  # 数据集分组，当数据集过大，工具一次性分析可能会卡死
func_level=True,  # 完整函数或者进行切片，这里建议使用完整函数
nodes_dim=505,  # 最大节点数
embed_dim=100,
vul_ratio=1,  # 漏洞:良性 样本分割比例
spgs_dir="./joern/repository/",
fpgs_dir="./joern/repository/",  # 图保存目录
ast_attr_path="./joern/files/our_map_all.txt",
gen_graph=False,  # 从头生成属性图，True 表示进行生成
with_load=False,  # 从目录加载属性图
gen_w2v=False,  # 根据数据集样本生成词嵌入
g2dataset=False,  # 将属性图生成图数据集
label_path="",
corpus_dir="./input/corpus/",
w2v_dir="./input/w2v/",
dataset_dir="./input/dataset/"  # 数据集原始路径
```
由于这里我们只需要得到属性图，所以只需打开 `gen_graph` 选项即可。
由属性图生成依赖预测、补丁预测子任务数据集的操作交由 `/dataset/data_preprocess.py` 完成。

### 系统运行

对于我们的系统（使用 CodeLlama），我们采用了它的官方实现版本  llama-recipes。 训练脚本位于 `/CodeLlama/configs/datasets.py`。 我们使用的是 alpaca 格式的数据集，其具体实现在 alpaca_dataset 类中。

在运行下面脚本之前请到 `/CodeLlama/configs` 目录下设置正确的模型、数据路径。

**微调**
```bash
cd CodeLlama
python finetuning.py \
    --use_peft \
    --model_name codellama/CodeLlama-13b-hf \
    --peft_method lora \
    --batch_size_training 32 \
    --val_batch_size 32 \
    --context_length 512 \
    --quantization \
    --num_epochs 3 \
    --output_dir codellama-13b-multi-r16
```

**推理**
```bash
cd CodeLlama
python inference-basic.py \
    --model_type codellama \
    --base_model codellama/CodeLlama-13b-hf \
    --tuned_model odellama-13b-multi-r16 \
    --data_file dataset/BigVul/test.jsonl
```
