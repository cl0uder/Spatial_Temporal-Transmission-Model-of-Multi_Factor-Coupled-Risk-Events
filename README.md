# 多因素耦合风险事件时空传导模型

## 项目介绍
这是一个基于多因素耦合的风险事件时空传导模型，集成了LSTM深度学习和马尔科夫链分析方法，用于风险预测感知。

## 新增功能：导入真实数据
现在模型支持导入CSV或Excel格式的真实数据文件，通过命令行参数进行配置。

## 数据格式要求
导入的真实数据文件需要包含以下列：
1. **时间列**：记录时间戳或时间序列信息
2. **空间列**：记录空间位置或区域标识
3. **风险因素列**：多个反映风险的指标数据
4. **风险得分列**（可选）：如果没有，模型会自动计算

## 使用方法

### 使用真实数据
程序使用真实数据运行，可直接运行以下命令使用示例数据：

```bash
python risk_prediction_model.py
```

系统会自动使用 `risk_data_example.csv` 文件作为数据源，默认列名设置如下：
- 时间列：`date`
- 空间列：`location`
- 风险因素列：`temperature,humidity,pressure,wind_speed`

如需使用自定义数据文件和列名，可以指定参数：

```bash
python risk_prediction_model.py --data-file 你的数据文件.csv --time-column 你的时间列名 --space-column 你的空间列名 --factor-columns 因素列1,因素列2,... [--risk-column 风险得分列名]
```

## 示例数据说明
项目中包含的`risk_data_example.csv`文件展示了数据的格式要求：
- date: 日期时间列
- location: 空间位置列
- 多个风险因素列
- 可选的risk_score列

## 模型说明
1. **LSTM模型**：用于学习和预测风险的时间序列特征
2. **马尔科夫链**：用于分析和预测风险状态的转移
3. **数据处理流程**：
   - 数据加载与验证
   - 按时间和空间分组
   - 构建滑动窗口的时间序列数据
   - 标准化处理
   - 模型训练与预测
   - 结果评估与可视化

## 输出结果
- 模型训练历史图：`lstm_training_history.png`
- 风险预测可视化结果：`risk_prediction_visualization.png`
- 控制台输出模型性能指标（MSE、准确率等）

## 依赖库
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow

## 扩展建议
1. 尝试不同的深度学习模型结构
2. 优化风险等级划分方法
3. 添加更多的可视化分析图表
4. 集成实时数据更新功能