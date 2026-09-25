# Schema Vocabulary — crypto perp 日频受控语义词表（本地版）

> SEMANTIC_PLAN 与 semantic_key 的唯一合法词源。**只允许用这里的 id**，
> 自由文本 id 会让 semantic_key 失去去重和血缘意义。
> 因子族与数据可得性见 `factor-families.md`；本表按五维重述。

## 组合规则

- Event 选 1；Context 选 0~1；Quality 选 0~3；Direction 选 1；Output 选 1
- semantic_key 格式：`event | context | quality1+quality2（排序）| direction | output`，
  无 context / quality 记 `-`，例：`vol_spike | funding_crowding | volume_confirm | reversal | rank`
- **compose_factor 例外**：组合 ≥2 个事件源时，Event 位写 `eventA+eventB`（排序），
  各成分的 qualities 合并（排序、去重，仍 ≤3 个核心）；compose 的 Direction/Output
  必须各成分一致，不一致说明组合假设本身没想清楚
- 每个 id 标注数据层：**T1=日频 OHLCV/quote_volume；T2=trade_count/taker_buy_*；
  T3=funding**（口径见 factor-families.md）

## Event（市场发生了什么）

| id | 含义 | 数据层 |
|---|---|---|
| `large_move` | 单周期大幅涨跌（方向由 Direction 决定） | T1 |
| `breakout` | 创 N 日新高/新低、突破近期区间 | T1 |
| `vol_spike` | 波动率骤升 | T1 |
| `vol_squeeze` | 波动率压缩至极低分位 | T1 |
| `volume_surge` | 放量 | T1 |
| `volume_dryup` | 缩量 | T1 |
| `pv_diverge` | 量价背离（量增价滞 / 价增量缩） | T1 |
| `path_noisy` | 路径杂乱（长上下影、高路径长度、低路径效率） | T1 |
| `illiquidity_level` | 非流动性水平状态（amihud 类冲击成本：\|ret\|/quote_volume 及"单位量能波动"变体，慢变 unsigned 特征载体，供 `premium` 方向收割非流动性溢价） | T1 |
| `activity_burst` | 交易活跃度阵发性（trade_count / quote_volume 日序列的变异系数/集中度，高=事件驱动投机，慢变 unsigned 特征，供 `premium` 方向） | T1/T2 |
| `chase_flow` | 追涨盘结构（日频口径：taker 主动买占比与当日收益同向=动量追逐盘，背离=逆向盘，慢变 unsigned 特征，供 `premium` 方向） | T2 |
| `close_location` | 日内收盘位置（CLV=(2C-H-L)/(H-L)，持续收高=尾盘承接/吸筹，持续收低=派发，慢变 unsigned 特征，供 `premium` 方向） | T1 |
| `range_level` | 日内振幅水平（(H-L)/C 类已实现波动口径，高=彩票/投机需求，慢变 unsigned 特征，供 `premium` 方向；与 vol_spike/vol_squeeze 的变化类用法不同质） | T1 |
| `taker_imbalance` | 主动买卖失衡（taker_buy_quote_volume/quote_volume 及其时序变化） | T2 |
| `trade_size_shift` | 单笔均额（quote_volume/trade_count）/ 交易频率结构突变 | T2 |
| `funding_extreme` | 资金费率极端化（水平/均值/滚动极端） | T3 |
| `volume_instability` | 成交额日序列时序离散度（CV 类），高=放量阵发/事件驱动交易，慢变特征；可拆上涨/下跌日符号分量，供 `premium` 方向；与 volume_surge/dryup 的水平变化用法不同质，与 activity_burst 同概念不同字段口径（quote_volume vs trade_count） | T1 |
| `activity_comovement` | 个币活动流与市场级活动流的时序联动度（如日 quote_volume 对市场总成交额的 corr；高=参与市场级活动波=被关注，低=neglect/死币；慢变 unsigned 特征，供 `premium` 方向；与 volume_instability 的自身离散度不同质——那是方差结构，这是共变结构） | T1 |
| `price_ma_divergence` | 价格多周期均线发散度（MA_w(close) 跨周期离散度/价格水平，高=趋势换挡/多周期分歧，低=多周期收敛盘整；慢变 unsigned 特征，供 `premium` 方向；与 breakout 的点位突破、large_move 的单期幅度不同质——这是均线族的结构离散） | T1 |
| `overnight_intraday_split` | 收益在日内(open→close)与隔夜(prev close→open)两时段间的结构分解（日内主导=散户情绪交易驱动，隔夜主导=知情资金驱动；与 large_move 的带符号总收益不同质——控制总收益后仍有截面区分度） | T1 |
| `tail_asymmetry` | 收益方差在上/下行的构成不对称（上行半方差占比，高=彩票型正偏度结构；与 vol_spike/vol_squeeze 的波动率水平变化、range_level 的振幅水平不同质——这是方差的方向构成；与偏度三阶矩口径不同，是二阶矩分解） | T1 |
| `euphoria_beta` | 个币收益对市场亢奋日（等权市场收益 top 分位日）的尾部弹性/条件 beta（高=博彩资金在狂欢时集中追捧的标的；与 large_move 的自身尾部不同质——这是协同尾部；与 activity_comovement 的活动流共变不同质——这是收益在条件子样本上的弹性） | T1 |
| `anchor_distance` | 价格相对 N 日长期高点的连续贴近度（close / rolling_max(close, N)，贴近=锚点下反应不足；与 breakout 的"穿越事件"不同质——George-Hwang 的核心发现是**贴近度本身**（无须突破）即预测收益，这是状态而非事件） | T1 |
| `beta_exposure` | 个币收益对市场收益（universe 等权）的滚动回归 beta 水平（Frazzini-Pedersen BAB：高 beta 被杠杆受限资金哄抬而高估，低 beta 低估，premium 方向低 beta 多/高 beta 空；与 euphoria_beta 的条件尾部弹性不同质——这是全样本无条件系统风险敞口；与 range_level/vol 族的自身波动不同质——这是对市场共变的斜率） | T1 |
| `adoption_trend` | log(quote_volume) 的时序趋势（滚动回归斜率或长短期均值对数差，高=真实采用度/资金关注度扩张；与 volume_surge 的单日放量事件、volume_instability 的离散度不同质——这是水平的慢变方向性漂移；与价格动量不同字段——量先行的采用叙事） | T1 |
| `liquidity_beta` | 个币收益对市场流动性冲击（universe 平均 amihud 的日变化）的滚动相关/载荷（Pastor-Stambaugh 流动性风险溢价：高载荷=收益在流动性恶化时同步缩水，需更高预期收益补偿；与 illiquidity_level 的自身非流动性水平不同质——这是对市场级流动性因子的敏感度；与 beta_exposure 的市场收益斜率不同质——冲击变量是流动性而非收益） | T1 |
| `market_coupling` | 个币收益与等权市场收益的滚动相关水平（高=与市场共舞的被关注主流币，低=neglect/异质定价币，低耦合溢价方向做多；与 activity_comovement 的活动流（成交额）共变不同字段——这是收益共变；与 beta_exposure 的共变斜率不同质——这是纯相关成分，剔除了波动率比） | T1 |
| `idio_vol` | 市场模型残差波动率（rolling 窗口内 ret_i 对 mkt 回归的残差 std，Ang et al. 2006 IVOL 折价：高特异波动=彩票型投机标的被高估，premium 方向做空；与 range_level/vol_spike 的总波动口径不同质——剔除了市场共变成分；与 market_coupling 是同一回归的互补输出（一个是相关、一个是残差离散度），语义上承接其反方向证据） | T1 |
| `idio_vol_spike` | 特异波动率的时序突变（当前 IVOL 相对其自身长期基线的偏离，高=该币彩票性正在被急剧点燃；与 vol_spike 的总波动骤升不同质——剔除了市场共变成分；与 idio_vol 的水平状态不同质——这是时序变化量，截面含义从'谁是彩票股'变成'谁的彩票性正在升温'） | T1 |
| `flow_path_efficiency` | 资金流（log quote_volume）时序路径的符号化方向效率（净位移/路径长度，Kaufman ER 移植到量维度；高正=单向稳步放量=知情吸筹无 churn，高负=单向稳步缩量；与 adoption_trend 的水平斜率不同质——斜率混合事件性放量尖峰，ER 只奖励路径持续性；与 volume_instability 的离散度不同质——那是幅度结构，这是方向路径结构；价格类路径效率已证与 price_momentum 换皮，本词条限定量维度） | T1 |
| `listing_recency` | 币安永续上市后的日历时间（面板内首个有效收盘距今的天数，面板起点 2023-07-05 左截断；新上市=上市 hype 未退+低流通/解锁抛压，Ritter IPO 长期弱势的 crypto 映射，premium 方向做多老币/做空新币；与 illiquidity_level 的交易成本结构不同质——这是生命周期阶段；与 adoption_trend 的量趋势不同质——这是日历时间而非流量） | T1 |
| `price_delay` | 个币收益对市场收益的信息扩散迟滞度（Hou-Moskowitz 2005：ret_i 对 mkt_t 与 mkt_{t-1..t-k} 回归，delay = 1 − R²(仅同期)/R²(含滞后)，高=价格发现滞后/被忽视，premium 方向做多高延迟；与 market_coupling 的同期相关水平不同质——这是滞后项的边际解释力占比，条件于同期联动） | T1 |
| `information_discreteness` | 信息到达的离散度（Da-Gurun-Warachka 2014 FIP：sign(窗口动量) × (%跌日 − %涨日)，平滑连续信息→frog-in-pan 反应不足→延续，突兀集中信息→一步到位无后续；与 large_move 的幅度、path_noisy 的路径长度不同质——这是符号频率结构，与动量幅度条件交互使用） | T1 |
| `vol_of_vol` | 已实现波动率的时序不确定性（短窗 rv 在长窗上的 std 或 CV，模糊厌恶/风险不确定性折价：高 VoV=风险本身不稳定，premium 方向做空；与 idio_vol/range_level 的波动率水平不同质——这是波动率的二阶时序结构） | T1 |
| `liquidation_cascade` | 杠杆清算瀑布（极端价格冲击 × 量能尖峰的条件共振：机械强平流主导的非信息性过冲，耗尽后回弹；与 large_move 的纯幅度、volume_surge 的纯放量不同质——这是两者交互的事件签名，reversal 方向收割过冲） | T1 |
| `liquidity_spiral` | 非流动性的时序恶化速度（短期 amihud 相对长期基线的对数变化率，高=资金撤离/Brunnermeier-Pedersen 流动性螺旋，continuation 方向做空恶化者；与 illiquidity_level 的水平状态不同质——这是变化率；与 liquidity_beta 的市场流动性冲击载荷不同质——这是自身流动性路径的时序恶化） | T1 |
| `implied_spread` | 收益负一阶自协方差隐含的往返价差（Roll 1984：spread=2√(−Cov(r_t,r_{t−1}))，买卖反弹的直接估计；与 illiquidity_level 的 Amihud 价格冲击不同质——那是单位成交额冲击成本，这是价差成本，两个独立摩擦分量；与 variance_ratio 共享负自相关物理来源但只用 lag-1 截断） | T1 |
| `nominal_price` | 名义价格水平（log close 的慢变水平，Birru-Wang 单位偏见：低名义价币被散户"便宜"幻觉超买高估，premium 方向做多高名义价/做空低名义价；与一切价格变化率/相对位置用法不同质——这是价格本身的绝对水平） | T1 |
| `coskewness` | 个币收益与市场收益平方的协偏度（Harvey-Siddique 2000：负协偏度=崩盘时联动放大，承担系统性尾部风险获得溢价；与 beta_exposure 的一阶斜率、market_coupling 的线性相关不同质——这是二阶非线性联动；与 tail_asymmetry 的自身方差构成不同质——这是与市场的联动形状） | T1 |
| `downside_coupling_asym` | 市场下跌日与上涨日子样本相关性的不对称差（Ang-Chen 2002：跌市相关性不对称升高=丧失分散化价值，premium 折价；与 market_coupling 的无条件相关不同质——恰恰是对称部分剔除后的差分；与 euphoria_beta 的单侧亢奋弹性不同质——这是两侧差分） | T1 |
| `weekend_exposure` | 收益在周末 vs 工作日的结构暴露差（crypto 24/7 交易但机构流量 5/7，周末散户主导+流动性枯竭：周末收益贡献高的币承载散户拥挤流量，premium 折价；日历维度，与 overnight_intraday_split 的日内/隔夜分解不同质——这是星期历结构） | T1 |
| `vol_term_structure` | 短期实现波动相对长期基线的对数比值（rv_5/rv_60，波动率聚类的状态偏离，高=风险状态正在抬升；与 vol_of_vol 的波动不稳定性（二阶导）不同质——这是一阶状态；与 vol_spike 的事件型骤升不同质——连续比值的慢变特征） | T1 |
| `vol_beta` | 个币实现波动对市场波动的时序敏感度（波动率的 beta，高=波动放大器承担二阶矩风险；与 beta_exposure 的收益对收益斜率不同质——这是二阶矩对二阶矩的载荷；与 vol_of_vol 的自身波动不稳定性不同质——这是与市场波动的共变） | T1 |
| `cost_basis_overhang` | 价格相对成交额加权成本基线的未实现盈亏（Grinblatt-Han 2005 处置效应：悬浮盈亏驱动持有者卖出/持有倾向，形成漂移结构；与 anchor_distance 的价格锚（52w 高点）不同质——这是持仓者 P&L 锚，行为主体是存量持有者；注意库内已有 CGO_Factor 递归参考价实现，本词条限指换手加权窗口均值口径变体，增量须由相关性门控自证） | T1 |
| `jump_intensity` | 收益绝对值超过 ±2.5σ 的跳跃频率（unsigned 尾部厚度，崩盘/暴涨双重跳跃风险存量的溢价补偿；与 max_lottery 的 signed 单期最大实现值（彩票需求）不同质——这是频率计数、无符号；与 tail_asymmetry 的方向性方差比值不同质） | T1 |
| `variance_ratio` | k 日收益方差与 k 倍 1 日收益方差之比（Lo-MacKinlay VR：VR<1=均值回复/买卖反弹结构，为流动性提供者库存风险的补偿，premium 方向；与 implied_spread 共享负自相关物理来源但使用完整自相关结构而非 lag-1 截断，两者互为近亲须探针分流） | T1 |
| `clientele_structure` | 个币收益对零售段指数（低成交额一半币等权）与大户段指数（高成交额币成交额加权）相关性的差分（Barber-Odean/Kumar-Lee 客户群定价：被零售段边际定价的币由情绪资金主导而高估折价；与 market_coupling 的全体等权相关不同质——BTC 仅占 EW 2%，这是定价者身份的结构性拆分） | T1 |
| `kyle_lambda` | 个币收益对净签名订单流（taker 净买占比）的滚动回归斜率（Kyle 1985：单位净流量的价格冲击，微观结构脆弱度；与 illiquidity_level 的 |ret|/总成交额不同质——分母是净签名流且保留方向结构；与 taker_imbalance 的流量水平不同质——这是冲击弹性） | T1/T2 |
| `corwin_schultz_spread` | 两日 high/low 区间反解的隐含价差（Corwin-Schultz 2012：价差使 high 偏高 low 偏低，单日内与跨日区间方差比含价差信息；与 implied_spread 的收益自相关口径、illiquidity_level 的成交额冲击口径都不同质——这是区间几何估计量；与 range_level 的振幅水平溢价不同质） | T1 |
| `vol_feedback` | 个币收益与自身实现波动变化的时序相关（Black/Christie 杠杆效应、Campbell-Hentschel 1992 波动反馈：跌时波动爆炸=困境敏感型，波动冲击本身压价的通道敏感度；与 idio_vol 的水平、vol_beta 的波动对波动共变、vol_of_vol 的不稳定性都不同质——这是收益-波动变化的协动结构） | T1 |
| `btc_anchor_coupling` | 个币收益与 BTCUSDT 收益的滚动相关（等权市场中 BTC 仅占 1/50，BTC 耦合几乎未被 market_coupling 覆盖：与 BTC 脱钩=独立叙事/特质风险补偿；与 idio_vol 的残差 std 是同一回归的不同输出维度，须探针分流） | T1 |
| `information_lumpiness` | \|ret\| 或 rv 的滞后 1~5 日自相关均值（Clark 1973 分布混合假说：波动聚类=信息到达成块性，阵发式信息环境被模糊厌恶折价；与 vol_of_vol 的波动不稳定**幅度**不同质——这是波动的**时间结构**/持续性） | T1 |
| `vwap_basis` | 收盘价相对当日量加权价（quote_volume/volume）的持续偏离（尾盘知情买压/结算压力签名；与 close_location 的区间几何 CLV 不同质——分母是成交量加权共识价而非日内区间；注意影线不对称恒等于 −CLV 已证冗余，本词条限定 VWAP 口径） | T1 |
| `beta_instability` | 滚动 beta 自身的时序 std（Epstein-Schneider 模糊厌恶：系统暴露时变的币投资者无法确定所持风险，折价；与 beta_exposure 的 beta 水平、vol_beta 的二阶矩共变都不同质——这是 beta 的时序二阶矩） | T1 |
| `underwater_supply` | 过去 N 日成交额中成交价高于当前价的占比（0~1 有界质量测度，处置效应供给侧表达：水下筹码在反弹至成本线时形成解套卖压；与 cost_basis_overhang/cgo 的价格比值（距离型）不同构造——这是质量占比型，cgo 家族最后一个独立构造） | T1 |
| `hurst_exponent` | 多视界方差比 VR(k) 的对数斜率估计的长记忆参数（H<0.5=均值回复结构；与 variance_ratio 单视界水平同族近亲——多视界曲率含不同信息但先验最低，探针 vs variance_ratio 缓存 >0.6 直接弃） | T1 |
| `spread_asymmetry` | 隐含价差在下跌日与上涨日子样本上的不对称（下行日 CS 均值 − 上行日 CS 均值：下跌时区间摩擦放大=抛压遇薄流动性的困境签名；与 tail_asymmetry 的收益方差方向构成不同字段口径——这是 H/L 区间几何估计量的符号条件分解；与 corwin_schultz_spread 的无条件水平不同质——这是条件不对称差分） | T1 |
| `overnight_gap_risk` | 开盘价相对前收的跳空幅度水平（mean_N(\|open/prev_close − 1\|)：24/7 市场仍存在跳空=流动性薄到无法平滑价格发现的不连续定价风险；与 jump_intensity 的 close-close 收益尾部计数不同质——跳空隔离了跨周期重新定价通道；与 overnight_intraday_split 的带符号收益分解不同质——这是 unsigned 幅度） | T1 |
| `volume_vol_elasticity` | 收益绝对值与成交额的时序耦合度（corr_N(\|ret\|, quote_volume)：分布混合假说的截面表达——高耦合=信息以事件波形式到达的散户赌场型场所，低耦合=连续双边流的机构型场所；与 information_lumpiness 的 \|ret\| 时间自相关不同质——那是单变量时间结构，这是跨矩耦合；与 illiquidity_level 的比率水平不同质——这是相关性结构） | T1 |

## Context（在什么状态下解释这个 event）

| id | 含义 | 数据层 |
|---|---|---|
| `vol_regime` | 高/低市场波动状态 | T1 |
| `btc_trend` | BTC 趋势状态（上涨/下跌/震荡） | T1 |
| `recent_extreme_zone` | 价格处于近期高/低位区 | T1 |
| `liquidity_bucket` | 流动性/amihud 分位桶 | T1 |
| `funding_crowding` | 截面 funding 分位（多空拥挤度） | T3 |

## Quality（什么证据确认/过滤这个 event，0~3 个）

| id | 含义 | 数据层 |
|---|---|---|
| `volume_confirm` | 量能确认 | T1 |
| `taker_confirm` | taker 流确认 | T2 |
| `funding_confirm` | funding 方向确认 | T3 |
| `multi_horizon_consistency` | 多时间尺度方向一致（多日窗口） | T1 |
| `persistence` | 条件持续 N 期才生效 | T1 |
| `path_cleanliness` | 路径干净（影线短、趋势平滑） | T1 |
| `breadth_confirm` | 截面广度确认 | T1 |
| `vol_squeeze_filter` | 波动压缩位置作驼峰过滤（event 转 quality 用法：极端压缩=死币、极端扩张=彩票，中段健康，逐币降权有毒 vol 状态） | T1 |
| `outlier_filter` | 剔除极端异常值 | T1 |
| `liquidity_filter` | 剔除低流动性标的 | T1 |

## Direction（对未来收益的方向假设）

| id | 含义 |
|---|---|
| `continuation` | 事件方向延续 |
| `reversal` | 事件方向反转 |
| `mean_reversion` | 向均衡/均值回归 |
| `relative_value` | 相对价差收敛 |
| `oscillation` | 区间震荡，两端反向 |
| `regime_split` | 按截面分区分方向：极端区走 continuation、中段区走 reversal（同因子双机制，用于"尾部延续/中段反转"共存结构） |
| `premium` | 特征溢价收割：无方向预测，符号由经济学先验固定（如彩票/尾部风险空、非流动/稳定多），用于 unsigned 截面特征因子 |

## Output（最终可交易表达）

| id | 含义 |
|---|---|
| `rank` | 截面 rank |
| `zscore` | 截面或时序 zscore |
| `bounded_score` | 有界连续分数（如 [-1,1]） |
| `event_decay` | 事件触发后按半衰期衰减 |
| `binary_condition` | 条件触发/未触发的 0-1（再截面化） |
| `residualized` | 对 beta/族因子残差化后的分数 |

## 已删除的词（原词表有、本地数据不支持）

以下 id 依赖本地不存在的字段，**禁止使用**：`positioning_buildup` /
`positioning_chase`（需 open_interest）、`basis_dislocation` / `basis_percentile`
（需 mark/index 价）、`mark_premium_confirm`（需 mark_close）、
`venue_share_shift`（需跨所数据）、`session` / `conditional_minutes` /
`intraday_structure` / `volume_clock_structure`（需分钟数据）、
`listing_tier`（无上市分层掩码）、`size_bucket`（市值不作信号字段）。

## 新增词条规则（唯一维护入口）

新 idea 需要新词时，**必须同时满足**才允许加词，且与本 skill 的改动同轮完成：

1. 数据可得性已确认（属于 T1/T2/T3 哪一层，字段须在 `DataProvider.list_datas()` 里）
2. 写出一句话机制含义（加进对应表）
3. 确认不是已有 id 的别名/换皮（如 `momentum_burst` ≈ `large_move` 就不加）

禁止：为单个因子临时造词不入表——下一次 semantic_key 对比立刻失效。
