import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class RiskPredictionSystem:
    def __init__(self):
        # 初始化参数
        self.num_factors = 5  # 风险因素数量
        self.num_time_steps = 5  # 时间步长（减小到5，以便生成更多的窗口）
        self.num_space_nodes = 5  # 空间节点数量
        self.risk_levels = {'低风险': 0, '中风险': 1, '高风险': 2}  # 风险等级映射
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.markov_transition_matrix = None
        self.output_dir = None
        self.run_count = None
        
    def setup_output_directory(self):
        """
        设置输出目录，创建plot_n文件夹
        """
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')
        
        # 确保image文件夹存在
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # 查找当前最大的运行次数
        max_count = 0
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('plot_'):
                try:
                    count = int(item[5:])  # 提取plot_后面的数字
                    if count > max_count:
                        max_count = count
                except ValueError:
                    continue
        
        # 计算新的运行次数
        self.run_count = max_count + 1
        
        # 创建新的文件夹
        self.output_dir = os.path.join(base_dir, f'plot_{self.run_count}')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        print(f"创建输出文件夹: {self.output_dir}")
        return self.output_dir
    
    def import_real_data(self, file_path, time_column, space_column, factor_columns, risk_column=None):
        """
        从CSV或Excel文件导入真实数据
        
        参数:
            file_path: 数据文件路径 (支持.csv和.xlsx格式)
            time_column: 时间戳列名
            space_column: 空间标识列名
            factor_columns: 风险因素列名列表
            risk_column: 风险标签/得分列名 (可选)
            
        返回:
            factors_data: 格式化后的风险因素数据 (样本数, 时间步长, 因素数)
            space_onehot: 空间特征的独热编码
            risk_labels: 风险等级标签
            risk_scores: 风险得分
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件扩展名选择读取方法
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}，仅支持.csv和.xlsx")
        except Exception as e:
            raise Exception(f"读取文件时出错: {str(e)}")
        
        # 检查必要的列是否存在
        required_columns = [time_column, space_column] + factor_columns
        if risk_column:
            required_columns.append(risk_column)
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"文件中缺少必要的列: {missing_columns}")
        
        # 打印调试信息
        print(f"数据文件包含 {len(df)} 行数据")
        if risk_column:
            print(f"风险得分列 ({risk_column}) 范围: [{df[risk_column].min():.2f}, {df[risk_column].max():.2f}]")
        
        # 更新参数
        self.num_factors = len(factor_columns)
        
        # 按时间和空间分组，构建时间序列数据
        # 首先按空间标识分组
        grouped = df.groupby(space_column)
        
        # 初始化结果数组
        all_factors_data = []
        all_space_onehot = []
        all_risk_scores = []
        
        # 为每个空间节点构建时间序列数据
        space_ids = list(grouped.groups.keys())
        self.num_space_nodes = len(space_ids)
        space_id_map = {space_id: idx for idx, space_id in enumerate(space_ids)}
        
        for space_id, group in grouped:
            # 按时间排序
            group_sorted = group.sort_values(by=time_column)
            
            # 提取风险因素数据
            factors_data = group_sorted[factor_columns].values
            
            # 提取风险得分（如果有）
            if risk_column:
                risk_scores = group_sorted[risk_column].values
            else:
                # 如果没有风险得分，使用因素的加权和作为得分
                weights = np.random.rand(self.num_factors)
                risk_scores = np.sum(factors_data * weights, axis=1)
            
            print(f"空间节点 {space_id} 包含 {len(factors_data)} 个时间点数据")
            print(f"空间节点 {space_id} 风险得分范围: [{np.min(risk_scores):.2f}, {np.max(risk_scores):.2f}]")
            
            # 构建时间窗口数据
            window_count = len(factors_data) - self.num_time_steps + 1
            print(f"空间节点 {space_id} 生成 {window_count} 个时间窗口")
            
            # 确保时间步长不会导致没有窗口
            if window_count < 1:
                # 如果时间步长太大，使用较小的时间步长
                adjusted_time_steps = max(1, len(factors_data) // 2)
                print(f"调整时间步长从 {self.num_time_steps} 到 {adjusted_time_steps}")
                window_count = len(factors_data) - adjusted_time_steps + 1
                for i in range(window_count):
                    window_factors = factors_data[i:i+adjusted_time_steps]
                    # 取窗口最后一个时间点的风险得分
                    window_risk_score = risk_scores[i+adjusted_time_steps-1]
                    
                    all_factors_data.append(window_factors)
                    all_risk_scores.append(window_risk_score)
                    
                    # 生成空间特征的独热编码
                    space_onehot = np.zeros(self.num_space_nodes)
                    space_onehot[space_id_map[space_id]] = 1
                    all_space_onehot.append(space_onehot)
            else:
                for i in range(window_count):
                    window_factors = factors_data[i:i+self.num_time_steps]
                    # 取窗口最后一个时间点的风险得分
                    window_risk_score = risk_scores[i+self.num_time_steps-1]
                    
                    all_factors_data.append(window_factors)
                    all_risk_scores.append(window_risk_score)
                    
                    # 生成空间特征的独热编码
                    space_onehot = np.zeros(self.num_space_nodes)
                    space_onehot[space_id_map[space_id]] = 1
                    all_space_onehot.append(space_onehot)
        
        # 转换为numpy数组
        factors_data = np.array(all_factors_data)
        space_onehot = np.array(all_space_onehot)
        risk_scores = np.array(all_risk_scores)
        
        print(f"总窗口数: {len(factors_data)}")
        if len(risk_scores) > 0:
            print(f"窗口风险得分范围: [{np.min(risk_scores):.2f}, {np.max(risk_scores):.2f}]")
        
        # 将风险得分映射到风险等级
        if len(risk_scores) > 0:
            # 使用分位数来确定风险等级的阈值
            q1, q3 = np.percentile(risk_scores, [33, 67])
            # 保存分位数阈值，用于预测时使用
            self.q1 = q1
            self.q3 = q3
            risk_labels = np.digitize(risk_scores, [q1, q3])
        else:
            risk_labels = np.array([])
            self.q1 = 0
            self.q3 = 0
        
        return factors_data, space_onehot, risk_labels, risk_scores
    
    def build_lstm_model(self):
        """
        构建LSTM模型用于时序风险预测
        """
        model = Sequential()
        # LSTM层捕获时间序列特征
        model.add(LSTM(64, input_shape=(self.num_time_steps, self.num_factors), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        # 输出层预测风险得分
        model.add(Dense(1, activation='linear'))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.lstm_model = model
        return model
    
    def train_lstm_model(self, x_train, y_train, epochs=50, batch_size=32):
        """
        训练LSTM模型
        """
        # 数据预处理
        x_reshaped = x_train.reshape(-1, self.num_factors)
        x_scaled = self.scaler.fit_transform(x_reshaped)
        x_train_scaled = x_scaled.reshape(-1, self.num_time_steps, self.num_factors)
        
        # 训练模型
        history = self.lstm_model.fit(
            x_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # 绘制训练历史
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel(' epoch')
        plt.ylabel('损失')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='训练MAE')
        plt.plot(history.history['val_mae'], label='验证MAE')
        plt.title('模型MAE')
        plt.xlabel('epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.tight_layout()
        
        # 保存到输出目录
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'lstm_training_history.png'))
        else:
            plt.savefig('lstm_training_history.png')
        
        plt.close()
        
        return history
    
    def build_markov_chain(self, risk_labels):
        """
        构建马尔科夫链模型用于风险状态转移分析
        """
        # 获取所有唯一的状态值并创建映射字典
        unique_states = sorted(set(risk_labels))
        state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
        num_states = len(unique_states)
        
        # 初始化转移矩阵
        transition_matrix = np.zeros((num_states, num_states))
        
        # 计算状态转移次数
        for i in range(len(risk_labels) - 1):
            current_state = risk_labels[i]
            next_state = risk_labels[i + 1]
            
            # 将状态值映射到矩阵索引
            current_idx = state_to_idx[current_state]
            next_idx = state_to_idx[next_state]
            
            transition_matrix[current_idx, next_idx] += 1
        
        # 归一化得到转移概率
        for i in range(num_states):
            if np.sum(transition_matrix[i]) > 0:
                transition_matrix[i] = transition_matrix[i] / np.sum(transition_matrix[i])
        
        self.markov_transition_matrix = transition_matrix
        self.state_to_idx = state_to_idx  # 保存状态到索引的映射
        return transition_matrix
    
    def predict_risk(self, factors_data):
        """
        预测风险：结合LSTM模型和马尔科夫链
        """
        # 使用LSTM模型预测风险得分
        x_reshaped = factors_data.reshape(-1, self.num_factors)
        x_scaled = self.scaler.transform(x_reshaped)
        x_input_scaled = x_scaled.reshape(-1, self.num_time_steps, self.num_factors)
        lstm_predictions = self.lstm_model.predict(x_input_scaled)
        
        # 将预测得分映射到风险等级
        # 使用与训练数据相同的分位数阈值来映射风险等级
        predicted_labels = np.digitize(lstm_predictions.flatten(), [self.q1, self.q3])
        
        # 使用马尔科夫链预测下一个状态
        next_state_predictions = []
        if hasattr(self, 'state_to_idx') and self.markov_transition_matrix is not None:
            # 获取索引到状态的反向映射
            idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
            
            for label in predicted_labels:
                # 确保标签在映射中存在，如果不存在则跳过
                if label in self.state_to_idx:
                    # 获取当前状态对应的矩阵索引
                    current_idx = self.state_to_idx[label]
                    # 根据当前状态和转移矩阵预测下一状态概率
                    next_state_probs = self.markov_transition_matrix[current_idx].copy()
                    
                    # 确保概率和为1（由于浮点数精度问题）
                    prob_sum = np.sum(next_state_probs)
                    if prob_sum > 0:
                        next_state_probs = next_state_probs / prob_sum
                    else:
                        # 如果所有概率都是0，使用均匀分布
                        next_state_probs = np.ones_like(next_state_probs) / len(next_state_probs)
                    
                    # 采样得到下一状态的索引
                    next_idx = np.random.choice(len(next_state_probs), p=next_state_probs)
                    # 将索引转换回原始状态值
                    next_state = idx_to_state[next_idx]
                    next_state_predictions.append(next_state)
        
        return lstm_predictions, predicted_labels, next_state_predictions
    
    def visualize_risk_prediction(self, risk_scores, predicted_scores, risk_labels, predicted_labels):
        """
        可视化风险预测结果
        """
        # 打印数据范围信息，用于调试
        print(f"真实风险得分范围: [{np.min(risk_scores):.2f}, {np.max(risk_scores):.2f}]")
        print(f"预测风险得分范围: [{np.min(predicted_scores):.2f}, {np.max(predicted_scores):.2f}]")
        
        plt.figure(figsize=(15, 10))
        
        # 1. 风险得分分布
        plt.subplot(2, 2, 1)
        # 动态调整bins数量，确保直方图能正确显示
        bins = min(20, len(np.unique(risk_scores)) // 2 + 1)
        if bins < 2:  # 确保至少有2个bin
            bins = 2
        
        plt.hist(risk_scores, bins=bins, alpha=0.5, label='真实风险得分')
        plt.hist(predicted_scores, bins=bins, alpha=0.5, label='预测风险得分')
        plt.title('风险得分分布')
        plt.xlabel('风险得分')
        plt.ylabel('频率')
        plt.legend()
        # 自动调整x轴范围，确保所有数据都能显示
        plt.autoscale(enable=True, axis='x', tight=True)
        
        # 2. 风险等级混淆矩阵
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        plt.subplot(2, 2, 2)
        
        # 获取唯一的风险标签
        unique_labels = sorted(set(risk_labels).union(set(predicted_labels)))
        
        # 根据实际的风险标签数量生成对应的中文标签
        label_map = {0: '低风险', 1: '中风险', 2: '高风险'}
        display_labels = [label_map.get(label, str(label)) for label in unique_labels]
        
        cm = confusion_matrix(risk_labels, predicted_labels, labels=unique_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=plt.gca())
        plt.title('风险等级预测混淆矩阵')
        
        # 3. 预测与真实值对比（前100个样本）
        plt.subplot(2, 2, 3)
        sample_count = min(100, len(risk_scores))
        plt.plot(risk_scores[:sample_count], 'b-', label='真实风险得分')
        plt.plot(predicted_scores[:sample_count], 'r--', label='预测风险得分')
        plt.title('风险得分预测与真实值对比')
        plt.xlabel('样本索引')
        plt.ylabel('风险得分')
        plt.legend()
        # 自动调整y轴范围，确保所有数据都能显示
        plt.autoscale(enable=True, axis='y', tight=True)
        
        # 4. 马尔科夫转移矩阵热图
        if self.markov_transition_matrix is not None:
            plt.subplot(2, 2, 4)
            num_states = self.markov_transition_matrix.shape[0]
            plt.imshow(self.markov_transition_matrix, cmap='hot', interpolation='nearest')
            plt.colorbar(label='转移概率')
            plt.title('马尔科夫链状态转移矩阵')
            
            # 获取状态到索引的映射，然后创建索引到状态的反向映射
            if hasattr(self, 'state_to_idx') and self.state_to_idx is not None:
                idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
                # 生成状态标签列表
                state_labels = []
                for i in range(num_states):
                    if i in idx_to_state:
                        # 根据状态值生成中文标签
                        if idx_to_state[i] == 0:
                            state_labels.append('低风险')
                        elif idx_to_state[i] == 1:
                            state_labels.append('中风险')
                        elif idx_to_state[i] == 2:
                            state_labels.append('高风险')
                        else:
                            state_labels.append(f'风险{idx_to_state[i]}')
                    else:
                        state_labels.append(f'状态{i}')
                plt.xticks(range(num_states), state_labels, rotation=45)
                plt.yticks(range(num_states), state_labels)
            else:
                plt.xticks(range(num_states), [f'状态{i}' for i in range(num_states)])
                plt.yticks(range(num_states), [f'状态{i}' for i in range(num_states)])
            
            # 在热图上显示数值
            for i in range(num_states):
                for j in range(num_states):
                    plt.text(j, i, f'{self.markov_transition_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black')
        
        plt.tight_layout()
        
        # 保存到输出目录
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'risk_prediction_visualization.png'))
        else:
            plt.savefig('risk_prediction_visualization.png')
        
        plt.close()
        
    def run_simulation(self, data_file_path, time_column, space_column, factor_columns, risk_column=None):
        """
        运行完整的风险预测模拟
        
        参数:
            data_file_path: 真实数据文件路径
            time_column: 时间戳列名
            space_column: 空间标识列名
            factor_columns: 风险因素列名列表
            risk_column: 风险标签/得分列名（可选）
        """
        if not data_file_path:
            raise ValueError("必须提供数据文件路径")
        if not time_column or not space_column or not factor_columns:
            raise ValueError("必须提供时间列、空间列和风险因素列")
            
        # 设置输出目录
        self.setup_output_directory()
            
        print(f"导入真实数据文件: {data_file_path}...")
        factors_data, space_onehot, risk_labels, risk_scores = self.import_real_data(
            data_file_path, time_column, space_column, factor_columns, risk_column
        )
        
        print("数据加载完成，样本数：", len(factors_data))
        print(f"风险等级分布：低风险: {np.sum(risk_labels == 0)}, 中风险: {np.sum(risk_labels == 1)}, 高风险: {np.sum(risk_labels == 2)}")
        
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test, labels_train, labels_test = train_test_split(
            factors_data, risk_scores, risk_labels, test_size=0.2, random_state=42
        )
        
        print("构建并训练LSTM模型...")
        self.build_lstm_model()
        self.train_lstm_model(x_train, y_train)
        
        print("构建马尔科夫链模型...")
        self.build_markov_chain(labels_train)
        
        print("预测测试集风险...")
        predicted_scores, predicted_labels, next_state_predictions = self.predict_risk(x_test)
        
        print("评估模型性能...")
        from sklearn.metrics import mean_squared_error, accuracy_score
        mse = mean_squared_error(y_test, predicted_scores)
        accuracy = accuracy_score(labels_test, predicted_labels)
        
        print(f"LSTM模型MSE: {mse:.4f}")
        print(f"风险等级预测准确率: {accuracy:.4f}")
        
        print("可视化预测结果...")
        self.visualize_risk_prediction(y_test, predicted_scores.flatten(), labels_test, predicted_labels)
        
        print("模拟完成！结果已保存为图片文件。")

# 主函数
if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='多因素耦合风险事件时空传导模型')
    parser.add_argument('--data-file', type=str, default='risk_data_example.csv', help='真实数据文件路径（默认：risk_data_example.csv）')
    parser.add_argument('--time-column', type=str, default='date', help='时间列名（默认：date）')
    parser.add_argument('--space-column', type=str, default='location', help='空间列名（默认：location）')
    parser.add_argument('--factor-columns', type=str, default='temperature,humidity,pressure,wind_speed,rainfall', help='风险因素列名，用逗号分隔（默认：temperature,humidity,pressure,wind_speed,rainfall）')
    parser.add_argument('--risk-column', type=str, default='risk_score', help='风险得分列名（默认：risk_score）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建风险预测系统实例
    risk_system = RiskPredictionSystem()
    
    # 使用真实数据运行
    print("使用真实数据运行模拟...")
    print(f"数据文件: {args.data_file}")
    print(f"时间列: {args.time_column}")
    print(f"空间列: {args.space_column}")
    print(f"风险因素列: {args.factor_columns}")
    
    # 解析风险因素列名
    factor_columns = args.factor_columns.split(',')
    # 运行模拟
    risk_system.run_simulation(
        data_file_path=args.data_file,
        time_column=args.time_column,
        space_column=args.space_column,
        factor_columns=factor_columns,
        risk_column=args.risk_column
    )