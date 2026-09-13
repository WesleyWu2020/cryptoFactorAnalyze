# Alpha191 因子生成提示词

来源：国泰君安《基于短周期价量特征的多因子选股体系》表6（因子明细）和附录2（变量及函数说明）。

说明：下面每个小节都可以单独复制给其他大模型使用。公式仅修复了 PDF 断行造成的英文单词空格问题，未主动改写原始公式逻辑。

## Alpha1

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha1 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE - OPEN) / OPEN)),6))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha1(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha1。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha2

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha2 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)),1))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha2(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha2。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha3

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha3 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha3(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha3。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha4

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha4 生成可运行、向量化、无未来函数的因子实现。

公式：
((((SUM(CLOSE,8) / 8) + STD(CLOSE,8)) < (SUM(CLOSE,2) / 2)) ? (-1 * 1) : (((SUM(CLOSE,2) / 2) < ((SUM(CLOSE,8) / 8) - STD(CLOSE,8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME / MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha4(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha4。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha5

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha5 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha5(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha5。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha6

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha6 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))),4)))* -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha6(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha6。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha7

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha7 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK(MAX((VWAP - CLOSE),3)) + RANK(MIN((VWAP - CLOSE),3))) * RANK(DELTA(VOLUME,3)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha7(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha7。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha8

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha8 生成可运行、向量化、无未来函数的因子实现。

公式：
RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)),4) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha8(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha8。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha9

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha9 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha9(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha9。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha10

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha10 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(MAX(((RET < 0) ? STD(RET,20) : CLOSE)^2),5))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha10(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha10。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha11

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha11 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha11(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha11。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha12

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha12 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK((OPEN - (SUM(VWAP,10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha12(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha12。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha13

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha13 生成可运行、向量化、无未来函数的因子实现。

公式：
(((HIGH * LOW)^0.5) - VWAP)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha13(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha13。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha14

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha14 生成可运行、向量化、无未来函数的因子实现。

公式：
CLOSE-DELAY(CLOSE,5)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha14(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha14。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha15

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha15 生成可运行、向量化、无未来函数的因子实现。

公式：
OPEN/DELAY(CLOSE,1)-1

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha15(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha15。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha16

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha16 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha16(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha16。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha17

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha17 生成可运行、向量化、无未来函数的因子实现。

公式：
RANK((VWAP - MAX(VWAP,15)))^DELTA(CLOSE,5)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha17(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha17。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha18

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha18 生成可运行、向量化、无未来函数的因子实现。

公式：
CLOSE/DELAY(CLOSE,5)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha18(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha18。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha19

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha19 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha19(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha19。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha20

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha20 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha20(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha20。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha21

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha21 生成可运行、向量化、无未来函数的因子实现。

公式：
REGBETA(MEAN(CLOSE,6),SEQUENCE(6))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha21(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha21。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha22

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha22 生成可运行、向量化、无未来函数的因子实现。

公式：
SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha22(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha22。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha23

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha23 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha23(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha23。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha24

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha24 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(CLOSE-DELAY(CLOSE,5),5,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha24(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha24。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha25

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha25 生成可运行、向量化、无未来函数的因子实现。

公式：
((-1 * RANK((DELTA(CLOSE,7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)),9)))))) * (1 + RANK(SUM(RET,250))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha25(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha25。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha26

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha26 生成可运行、向量化、无未来函数的因子实现。

公式：
((((SUM(CLOSE,7) / 7) - CLOSE)) + ((CORR(VWAP,DELAY(CLOSE,5),230))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha26(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha26。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha27

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha27 生成可运行、向量化、无未来函数的因子实现。

公式：
WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha27(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha27。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha28

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha28 生成可运行、向量化、无未来函数的因子实现。

公式：
3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha28(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha28。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha29

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha29 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha29(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha29。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha30

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha30 生成可运行、向量化、无未来函数的因子实现。

公式：
WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML,60))^2,20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha30(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha30。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha31

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha31 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha31(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha31。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha32

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha32 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),3)),3))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha32(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha32。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha33

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha33 生成可运行、向量化、无未来函数的因子实现。

公式：
((((-1 * TSMIN(LOW,5)) + DELAY(TSMIN(LOW,5),5)) * RANK(((SUM(RET,240) - SUM(RET,20)) / 220))) * TSRANK(VOLUME,5))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha33(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha33。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha34

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha34 生成可运行、向量化、无未来函数的因子实现。

公式：
MEAN(CLOSE,12)/CLOSE

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha34(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha34。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha35

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha35 生成可运行、向量化、无未来函数的因子实现。

公式：
(MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),RANK(DECAYLINEAR(CORR((VOLUME),((OPEN * 0.65) + (OPEN *0.35)),17),7))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha35(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha35。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha36

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha36 生成可运行、向量化、无未来函数的因子实现。

公式：
RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP)),6),2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha36(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha36。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha37

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha37 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * RANK(((SUM(OPEN,5) * SUM(RET,5)) - DELAY((SUM(OPEN,5) * SUM(RET,5)),10))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha37(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha37。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha38

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha38 生成可运行、向量化、无未来函数的因子实现。

公式：
(((SUM(HIGH,20) / 20) < HIGH) ? (-1 * DELTA(HIGH,2)) : 0)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha38(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha38。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha39

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha39 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK(DECAYLINEAR(DELTA((CLOSE),2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),SUM(MEAN(VOLUME,180),37),14),12))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha39(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha39。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha40

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha40 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha40(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha40。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha41

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha41 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(MAX(DELTA((VWAP),3),5))* -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha41(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha41。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha42

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha42 生成可运行、向量化、无未来函数的因子实现。

公式：
((-1 * RANK(STD(HIGH,10))) * CORR(HIGH,VOLUME,10))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha42(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha42。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha43

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha43 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha43(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha43。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha44

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha44 生成可运行、向量化、无未来函数的因子实现。

公式：
(TSRANK(DECAYLINEAR(CORR(((LOW)),MEAN(VOLUME,10),7),6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3),10),15))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha44(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha44。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha45

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha45 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))),1)) * RANK(CORR(VWAP,MEAN(VOLUME,150),15)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha45(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha45。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha46

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha46 生成可运行、向量化、无未来函数的因子实现。

公式：
(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha46(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha46。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha47

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha47 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha47(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha47。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha48

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha48 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE,1))) + SIGN((DELAY(CLOSE,1) - DELAY(CLOSE,2)))) + SIGN((DELAY(CLOSE,2) - DELAY(CLOSE,3)))))) * SUM(VOLUME,5)) / SUM(VOLUME,20))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha48(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha48。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha49

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha49 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha49(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha49。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha50

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha50 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0: MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha50(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha50。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha51

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha51 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha51(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha51。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha52

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha52 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)* 100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha52(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha52。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha53

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha53 生成可运行、向量化、无未来函数的因子实现。

公式：
COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha53(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha53。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha54

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha54 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE,OPEN,10)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha54(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha54。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha55

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha55 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha55(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha55。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha56

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha56 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK((OPEN - TSMIN(OPEN,12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2),19),SUM(MEAN(VOLUME,40),19),13))^5)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha56(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha56。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha57

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha57 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha57(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha57。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha58

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha58 生成可运行、向量化、无未来函数的因子实现。

公式：
COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha58(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha58。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha59

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha59 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha59(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha59。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha60

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha60 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha60(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha60。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha61

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha61 生成可运行、向量化、无未来函数的因子实现。

公式：
(MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80),8)),17))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha61(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha61。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha62

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha62 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * CORR(HIGH,RANK(VOLUME),5))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha62(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha62。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha63

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha63 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha63(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha63。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha64

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha64 生成可运行、向量化、无未来函数的因子实现。

公式：
(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha64(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha64。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha65

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha65 生成可运行、向量化、无未来函数的因子实现。

公式：
MEAN(CLOSE,6)/CLOSE

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha65(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha65。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha66

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha66 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha66(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha66。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha67

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha67 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha67(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha67。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha68

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha68 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha68(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha68。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha69

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha69 生成可运行、向量化、无未来函数的因子实现。

公式：
(SUM(DTM,20)>SUM(DBM,20)?(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):(SUM(DTM,20)=SUM(DBM,20)? 0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha69(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha69。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha70

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha70 生成可运行、向量化、无未来函数的因子实现。

公式：
STD(AMOUNT,6)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha70(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha70。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha71

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha71 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha71(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha71。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha72

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha72 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha72(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha72。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha73

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha73 生成可运行、向量化、无未来函数的因子实现。

公式：
((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE),VOLUME,10),16),4),5) - RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha73(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha73。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha74

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha74 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)),20),SUM(MEAN(VOLUME,40),20),7)) + RANK(CORR(RANK(VWAP),RANK(VOLUME),6)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha74(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha74。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha75

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha75 生成可运行、向量化、无未来函数的因子实现。

公式：
COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha75(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha75。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha76

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha76 生成可运行、向量化、无未来函数的因子实现。

公式：
STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha76(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha76。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha77

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha77 生成可运行、向量化、无未来函数的因子实现。

公式：
MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)),20)),RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2),MEAN(VOLUME,40),3),6)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha77(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha77。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha78

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha78 生成可运行、向量化、无未来函数的因子实现。

公式：
((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha78(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha78。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha79

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha79 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha79(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha79。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha80

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha80 生成可运行、向量化、无未来函数的因子实现。

公式：
(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha80(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha80。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha81

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha81 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(VOLUME,21,2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha81(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha81。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha82

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha82 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha82(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha82。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha83

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha83 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),5)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha83(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha83。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha84

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha84 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha84(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha84。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha85

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha85 生成可运行、向量化、无未来函数的因子实现。

公式：
(TSRANK((VOLUME / MEAN(VOLUME,20)),20) * TSRANK((-1 * DELTA(CLOSE,7)),8))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha85(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha85。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha86

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha86 生成可运行、向量化、无未来函数的因子实现。

公式：
((0.25 < (((DELAY(CLOSE,20) - DELAY(CLOSE,10)) / 10) - ((DELAY(CLOSE,10) - CLOSE) / 10))) ? (-1 * 1) : (((((DELAY(CLOSE,20) - DELAY(CLOSE,10)) / 10) - ((DELAY(CLOSE,10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) * (CLOSE - DELAY(CLOSE,1)))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha86(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha86。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha87

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha87 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK(DECAYLINEAR(DELTA(VWAP,4),7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))),11),7)) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha87(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha87。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha88

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha88 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha88(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha88。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha89

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha89 生成可运行、向量化、无未来函数的因子实现。

公式：
2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha89(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha89。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha90

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha90 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(CORR(RANK(VWAP),RANK(VOLUME),5)) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha90(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha90。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha91

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha91 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK((CLOSE - MAX(CLOSE,5)))*RANK(CORR((MEAN(VOLUME,40)),LOW,5))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha91(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha91。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha92

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha92 生成可运行、向量化、无未来函数的因子实现。

公式：
(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15)) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha92(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha92。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha93

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha93 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha93(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha93。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha94

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha94 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha94(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha94。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha95

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha95 生成可运行、向量化、无未来函数的因子实现。

公式：
STD(AMOUNT,20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha95(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha95。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha96

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha96 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha96(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha96。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha97

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha97 生成可运行、向量化、无未来函数的因子实现。

公式：
STD(VOLUME,10)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha97(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha97。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha98

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha98 生成可运行、向量化、无未来函数的因子实现。

公式：
((((DELTA((SUM(CLOSE,100) / 100),100) / DELAY(CLOSE,100)) < 0.05) || ((DELTA((SUM(CLOSE,100) / 100),100) / DELAY(CLOSE,100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE,100))) : (-1 * DELTA(CLOSE,3)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha98(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha98。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha99

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha99 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),5)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha99(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha99。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha100

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha100 生成可运行、向量化、无未来函数的因子实现。

公式：
STD(VOLUME,20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha100(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha100。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha101

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha101 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),RANK(VOLUME),11))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha101(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha101。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha102

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha102 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha102(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha102。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha103

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha103 生成可运行、向量化、无未来函数的因子实现。

公式：
((20-LOWDAY(LOW,20))/20)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha103(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha103。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha104

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha104 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * (DELTA(CORR(HIGH,VOLUME,5),5) * RANK(STD(CLOSE,20))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha104(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha104。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha105

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha105 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * CORR(RANK(OPEN),RANK(VOLUME),10))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha105(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha105。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha106

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha106 生成可运行、向量化、无未来函数的因子实现。

公式：
CLOSE-DELAY(CLOSE,20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha106(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha106。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha107

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha107 生成可运行、向量化、无未来函数的因子实现。

公式：
(((-1 * RANK((OPEN - DELAY(HIGH,1)))) * RANK((OPEN - DELAY(CLOSE,1)))) * RANK((OPEN - DELAY(LOW,1))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha107(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha107。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha108

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha108 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK((HIGH - MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha108(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha108。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha109

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha109 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha109(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha109。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha110

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha110 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha110(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha110。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha111

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha111 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha111(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha111。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha112

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha112 生成可运行、向量化、无未来函数的因子实现。

公式：
(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha112(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha112。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha113

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha113 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * ((RANK((SUM(DELAY(CLOSE,5),20) / 20)) * CORR(CLOSE,VOLUME,2)) * RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha113(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha113。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha114

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha114 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE,5) / 5)),2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE,5) / 5)) / (VWAP - CLOSE)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha114(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha114。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha115

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha115 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH + LOW) / 2),4),TSRANK(VOLUME,10),7)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha115(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha115。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha116

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha116 生成可运行、向量化、无未来函数的因子实现。

公式：
REGBETA(CLOSE,SEQUENCE,20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha116(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha116。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha117

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha117 生成可运行、向量化、无未来函数的因子实现。

公式：
((TSRANK(VOLUME,32) * (1 - TSRANK(((CLOSE + HIGH) - LOW),16))) * (1 - TSRANK(RET,32)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha117(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha117。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha118

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha118 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha118(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha118。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha119

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha119 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha119(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha119。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha120

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha120 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha120(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha120。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha121

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha121 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK((VWAP - MIN(VWAP,12)))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3)) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha121(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha121。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha122

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha122 生成可运行、向量化、无未来函数的因子实现。

公式：
(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha122(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha122。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha123

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha123 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK(CORR(SUM(((HIGH + LOW) / 2),20),SUM(MEAN(VOLUME,60),20),9)) < RANK(CORR(LOW,VOLUME,6))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha123(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha123。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha124

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha124 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha124(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha124。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha125

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha125 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(DECAYLINEAR(CORR((VWAP),MEAN(VOLUME,80),17),20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)),3),16)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha125(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha125。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha126

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha126 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE+HIGH+LOW)/3

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha126(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha126。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha127

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha127 生成可运行、向量化、无未来函数的因子实现。

公式：
(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha127(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha127。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha128

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha128 生成可运行、向量化、无未来函数的因子实现。

公式：
100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha128(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha128。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha129

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha129 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha129(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha129。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha130

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha130 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2),MEAN(VOLUME,40),9),10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha130(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha130。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha131

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha131 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(DELAT(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha131(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha131。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha132

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha132 生成可运行、向量化、无未来函数的因子实现。

公式：
MEAN(AMOUNT,20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha132(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha132。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha133

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha133 生成可运行、向量化、无未来函数的因子实现。

公式：
((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha133(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha133。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha134

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha134 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha134(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha134。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha135

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha135 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha135(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha135。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha136

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha136 生成可运行、向量化、无未来函数的因子实现。

公式：
((-1 * RANK(DELTA(RET,3))) * CORR(OPEN,VOLUME,10))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha136(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha136。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha137

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha137 生成可运行、向量化、无未来函数的因子实现。

公式：
16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha137(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha137。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha138

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha138 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))),3),20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5),19),16),7)) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha138(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha138。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha139

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha139 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1 * CORR(OPEN,VOLUME,10))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha139(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha139。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha140

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha140 生成可运行、向量化、无未来函数的因子实现。

公式：
MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))),8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha140(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha140。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha141

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha141 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))* -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha141(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha141。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha142

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha142 生成可运行、向量化、无未来函数的因子实现。

公式：
(((-1 * RANK(TSRANK(CLOSE,10))) * RANK(DELTA(DELTA(CLOSE,1),1))) * RANK(TSRANK((VOLUME /MEAN(VOLUME,20)),5)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha142(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha142。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha143

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha143 生成可运行、向量化、无未来函数的因子实现。

公式：
CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha143(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha143。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha144

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha144 生成可运行、向量化、无未来函数的因子实现。

公式：
SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha144(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha144。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha145

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha145 生成可运行、向量化、无未来函数的因子实现。

公式：
(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha145(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha145。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha146

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha146 生成可运行、向量化、无未来函数的因子实现。

公式：
MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60);

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha146(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha146。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha147

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha147 生成可运行、向量化、无未来函数的因子实现。

公式：
REGBETA(MEAN(CLOSE,12),SEQUENCE(12))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha147(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha147。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha148

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha148 生成可运行、向量化、无未来函数的因子实现。

公式：
((RANK(CORR((OPEN),SUM(MEAN(VOLUME,60),9),6)) < RANK((OPEN - TSMIN(OPEN,14)))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha148(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha148。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha149

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha149 生成可运行、向量化、无未来函数的因子实现。

公式：
REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha149(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha149。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha150

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha150 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE+HIGH+LOW)/3*VOLUME

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha150(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha150。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha151

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha151 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(CLOSE-DELAY(CLOSE,20),20,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha151(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha151。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha152

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha152 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY (CLOSE,9),1),9,1),1),26),9,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha152(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha152。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha153

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha153 生成可运行、向量化、无未来函数的因子实现。

公式：
(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha153(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha153。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha154

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha154 生成可运行、向量化、无未来函数的因子实现。

公式：
(((VWAP - MIN(VWAP,16))) < (CORR(VWAP,MEAN(VOLUME,180),18)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha154(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha154。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha155

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha155 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha155(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha155。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha156

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha156 生成可运行、向量化、无未来函数的因子实现。

公式：
(MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)),2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1),3))) * -1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha156(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha156。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha157

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha157 生成可运行、向量化、无未来函数的因子实现。

公式：
(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1),5))))),2),1)))),1),5) + TSRANK(DELAY((-1 * RET),6),5))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha157(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha157。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha158

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha158 生成可运行、向量化、无未来函数的因子实现。

公式：
((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha158(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha158。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha159

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha159 生成可运行、向量化、无未来函数的因子实现。

公式：
((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6) *12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha159(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha159。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha160

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha160 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha160(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha160。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha161

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha161 生成可运行、向量化、无未来函数的因子实现。

公式：
MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha161(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha161。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha162

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha162 生成可运行、向量化、无未来函数的因子实现。

公式：
(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha162(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha162。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha163

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha163 生成可运行、向量化、无未来函数的因子实现。

公式：
RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha163(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha163。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha164

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha164 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha164(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha164。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha165

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha165 生成可运行、向量化、无未来函数的因子实现。

公式：
MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha165(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha165。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha166

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha166 生成可运行、向量化、无未来函数的因子实现。

公式：
-20* (20-1) ^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha166(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha166。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha167

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha167 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha167(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha167。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha168

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha168 生成可运行、向量化、无未来函数的因子实现。

公式：
(-1*VOLUME/MEAN(VOLUME,20))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha168(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha168。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha169

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha169 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha169(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha169。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha170

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha170 生成可运行、向量化、无未来函数的因子实现。

公式：
((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH,5) / 5))) - RANK((VWAP - DELAY(VWAP,5))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha170(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha170。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha171

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha171 生成可运行、向量化、无未来函数的因子实现。

公式：
((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha171(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha171。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha172

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha172 生成可运行、向量化、无未来函数的因子实现。

公式：
MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha172(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha172。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha173

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha173 生成可运行、向量化、无未来函数的因子实现。

公式：
3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2);

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha173(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha173。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha174

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha174 生成可运行、向量化、无未来函数的因子实现。

公式：
SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha174(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha174。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha175

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha175 生成可运行、向量化、无未来函数的因子实现。

公式：
MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha175(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha175。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha176

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha176 生成可运行、向量化、无未来函数的因子实现。

公式：
CORR(RANK(((CLOSE - TSMIN(LOW,12)) / (TSMAX(HIGH,12) - TSMIN(LOW,12)))),RANK(VOLUME),6)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha176(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha176。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha177

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha177 生成可运行、向量化、无未来函数的因子实现。

公式：
((20-HIGHDAY(HIGH,20))/20)*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha177(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha177。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha178

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha178 生成可运行、向量化、无未来函数的因子实现。

公式：
(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha178(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha178。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha179

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha179 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(CORR(VWAP,VOLUME,4)) *RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha179(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha179。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha180

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha180 生成可运行、向量化、无未来函数的因子实现。

公式：
((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE,7)),60)) * SIGN(DELTA(CLOSE,7)) : (-1 * VOLUME)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha180(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha180。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha181

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha181 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha181(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha181。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha182

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha182 生成可运行、向量化、无未来函数的因子实现。

公式：
COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha182(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha182。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha183

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha183 生成可运行、向量化、无未来函数的因子实现。

公式：
MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha183(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha183。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha184

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha184 生成可运行、向量化、无未来函数的因子实现。

公式：
(RANK(CORR(DELAY((OPEN - CLOSE),1),CLOSE,200)) + RANK((OPEN - CLOSE)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha184(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha184。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha185

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha185 生成可运行、向量化、无未来函数的因子实现。

公式：
RANK((-1 * ((1 - (OPEN / CLOSE))^2)))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha185(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha185。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha186

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha186 生成可运行、向量化、无未来函数的因子实现。

公式：
(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha186(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha186。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha187

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha187 生成可运行、向量化、无未来函数的因子实现。

公式：
SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha187(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha187。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha188

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha188 生成可运行、向量化、无未来函数的因子实现。

公式：
((HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha188(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha188。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha189

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha189 生成可运行、向量化、无未来函数的因子实现。

公式：
MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha189(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha189。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha190

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha190 生成可运行、向量化、无未来函数的因子实现。

公式：
LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)- 1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha190(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha190。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```

## Alpha191

```text
你是量化因子工程师。请严格按照下面的原始公式，为 Alpha191 生成可运行、向量化、无未来函数的因子实现。

公式：
((CORR(MEAN(VOLUME,20),LOW,5) + ((HIGH + LOW) / 2)) - CLOSE)

数据和函数约定：
输入为日频股票行情面板数据，至少包含 OPEN、HIGH、LOW、CLOSE、VWAP、VOLUME、AMOUNT、RET；若公式使用 BANCHMARKINDEXCLOSE/BANCHMARKINDEXOPEN、MKT/SMB/HML、SELF、DTM、DBM、TR、HD、LD 等变量，请先按定义派生。函数约定：RANK 为同一交易日股票截面升序排名；DELAY/DELTA/STD/MEAN/SUM/CORR/COVIANCE/TSMIN/TSMAX/TSRANK/PROD/COUNT/REGBETA/REGRESI/SMA/WMA/DECAYLINEAR/SUMIF/FILTER/HIGHDAY/LOWDAY/SUMAC 为按单只股票时间序列进行滚动、滞后或筛选计算；A?B:C 为逐元素条件表达式。

输出要求：
1. 先用中文拆解公式的计算步骤，并指出需要的输入字段和中间变量。
2. 生成 Python/pandas 代码，函数名为 calc_alpha191(df)，支持多股票、多日期面板数据；滚动窗口必须按股票分组计算，截面排名按交易日分组计算。
3. 处理缺失值、窗口不足、除零、停牌/无成交、极端值，并说明处理方式。
4. 附一个最小测试样例，验证输出列名为 Alpha191。
5. 不要改写因子经济含义；若发现原始公式疑似拼写或排版错误，请保留原式实现并在备注中标出。
```
