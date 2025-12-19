CAIL2018_textcls 项目 README

基于预训练语言模型的中文法律文本多任务分类：罪名（Accusation）/ 法条（Article）/ 刑期区间（Term）
已完成：RoBERTa-wwm-ext 全流程训练+测试+FastAPI+网页；Lawformer 半程训练（较慢）
最终测试口径：test.py（写 preds + 计算 metrics），阈值 threshold=0.2

1. 项目功能概览

数据处理

切分训练/验证：split_train_valid.py

构建标签映射：src/data/build_label_map.py → first_stage/label_map.json

数据管道自检：src/data/quick_check_dataset.py

模型训练

主训练脚本：src/train.py

支持模型：

hfl/chinese-roberta-wwm-ext（已跑通并产出 best）

thunlp/Lawformer（训练慢，半程）

测试与预测

最终测试脚本：src/test.py（输出 test_pred.jsonl + test_metrics.json）

单条预测示例（产品化）：FastAPI + 网页端

部署

推理服务：src/api_app.py（FastAPI）

前端网页：一个输入框 + 按钮调用 /predict（你本地已接好）

2. 目录结构
CAIL2018_small/
├─ first_stage/
│  ├─ train.json
│  ├─ test.json
│  ├─ train_split.json
│  ├─ valid_split.json
│  └─ label_map.json
│
├─ outputs/
│  ├─ exp_roberta_baseline/
│  │  ├─ config.json
│  │  ├─ best/
│  │  │  ├─ encoder/              # fine-tuned encoder（model.safetensors + config.json）
│  │  │  ├─ tokenizer/            # tokenizer 文件
│  │  │  ├─ multitask_head.pt     # 多任务头参数
│  │  │  └─ best_metrics.json     # 最佳验证指标记录
│  │  ├─ test_pred.jsonl          # test.py 生成：逐行预测结果
│  │  └─ test_metrics.json        # test.py 生成：测试指标
│  │
│  └─ exp_lawformer/
│     └─ best/ ...（若训练到 best 会生成同样结构）
│
└─ src/
   ├─ data/
   │  ├─ dataset.py
   │  ├─ build_label_map.py
   │  └─ quick_check_dataset.py
   ├─ labels/
   │  └─ term_bins.py
   ├─ models/
   │  └─ multitask_head.py
   ├─ train.py
   ├─ test.py
   ├─ predict.py
   └─ api_app.py

3. 环境与依赖安装（CUDA 12.6）

建议你用 venv：

python -m venv .venv
# Windows
.\.venv\Scripts\activate


安装 PyTorch（CUDA 12.6 对应 whl 通常用 cu126）：

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


再装其余依赖：

pip install transformers accelerate datasets sentencepiece tqdm numpy fastapi uvicorn pydantic


如果你遇到 HuggingFace 下载慢：可以设置镜像/代理，或保持当前方式即可。

4. 数据准备
4.1 切分 train/valid
python split_train_valid.py


你已得到：

first_stage/train_split.json

first_stage/valid_split.json

4.2 构建 label_map.json
python src/data/build_label_map.py


输出：

first_stage/label_map.json

标签规模：罪名 202 类、法条 183 类、刑期 9 桶（term_bins）

4.3 数据自检（推荐）
python src/data/quick_check_dataset.py


确认 batch 维度：

labels_accusation: [B,202]

labels_articles: [B,183]

labels_term: [B]

5. 训练
5.1 训练 RoBERTa-wwm-ext（已跑通）

src/train.py 中使用：

model_name = "hfl/chinese-roberta-wwm-ext"

max_len = 256

训练完成后会保存：

outputs/exp_roberta_baseline/best/
  encoder/
  tokenizer/
  multitask_head.pt
  best_metrics.json


运行：

python src/train.py

5.2 训练 Lawformer（训练慢，半程）

把 model_name 改为：

model_name = "thunlp/Lawformer"

输出目录建议改成：outputs/exp_lawformer/

运行：

python src/train.py


Lawformer 训练慢是正常现象（Longformer 结构 + attention_window 对齐/长序列开销更大）。你现在的算力方案下建议先以 RoBERTa 作为主结果提交。

6. 最终测试（推荐用 test.py 作为报告口径）

你已确定：最终测试口径使用 test.py（写 preds + metrics），threshold=0.2。

运行：

python src/test.py


输出文件：

outputs/exp_roberta_baseline/test_pred.jsonl（逐行 JSON 的预测结果）

outputs/exp_roberta_baseline/test_metrics.json（测试指标）

你给出的 RoBERTa 最终 test 指标（threshold=0.2）：

accusation_micro_f1 ≈ 0.8505

article_micro_f1 ≈ 0.8409

term_acc ≈ 0.6646

score ≈ 0.7853

7. 产品化推理（FastAPI + 网页）
7.1 启动 API
python -m uvicorn src.api_app:app --host 127.0.0.1 --port 8000


打开：

http://127.0.0.1:8000/docs

接口通常包含：

POST /predict：输入 fact 文本，返回

罪名 Top5

法条 Top5

刑期区间（term_bin → term_bins）

7.2 网页端

你写的网页通过 fetch("/predict") 调用即可，效果就是“输入一段文本 → 返回罪名/法条/刑期”，非常适合演示与写报告。

8. 输出解释（best/ 目录里有什么）

以 outputs/exp_roberta_baseline/best/ 为例：

encoder/：fine-tuned 后的编码器权重（RoBERTa 本体参数）

tokenizer/：分词器文件，保证线上/线下一致

multitask_head.pt：三任务分类头参数（罪名/法条/刑期）

best_metrics.json：验证集最优指标记录（用于报告）
