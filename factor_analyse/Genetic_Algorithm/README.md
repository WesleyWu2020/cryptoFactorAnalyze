# Genetic Algorithm Factor Mining

在本目录下用遗传算法（GA）自动生成“因子表达式”，并用 **IC + 分组多空收益** 加权作为适应度（fitness）进行进化。

## 关键设计

- **无未来函数**：表达式只允许使用当前及过去数据（rolling / shift(+k)）。
- **成分股可用性**：用 `index_cache_equal_weight_30_50.csv` 扩展到所有交易日后，对每个日期仅在可用 token 上计算与评估。
- **适应度（C）**：`fitness = w_ic * IC_IR + w_ls * LS_IR - complexity_penalty * expr_size - missing_penalty`。

## 运行

在项目根目录（Crypto）下执行：

```bash
python factor_analyse/Genetic_Algorithm/run_ga_factor_mining.py \
  --rebalance-period 10 \
  --top-n 50 \
  --generations 30 \
  --population 80
```

输出：
- 控制台逐代打印 best expression 与指标
- 在 `data/factor_data/` 下保存最优表达式对应的因子数据（`ga_expr_*` 前缀）

## 你可能想改的参数

- `--ic-weight / --ls-weight`：IC 与多空收益的权重
- `--group-n`：分组数（默认 5）
- `--max-depth / --max-nodes`：表达式复杂度上限
