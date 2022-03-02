# GoogleBrain-Ventilator-Pressure-Prediction Top-5%-Solution



## 比赛介绍

本次竞赛是由谷歌大脑举办的，参赛者将模拟一个连接到镇静病人肺部的呼吸机，并且根据呼吸机和肺部的状态来预测气道内的压力值。如果成功，我们将有助于开发新的控制呼吸机的方法，并减少开发成本。这将为适应病人的算法铺平道路，并在新冠时期及以后减少临床医生的负担。因此，呼吸机治疗可能会变得更加广泛，以帮助病人呼吸。

- 竞赛类型：本次竞赛属于时序表格数据，所以推荐使用时序相关的模型（例如LSTM，Transformer等）
- 评估标准：公式为：**|X-Y|**  ，以每次呼吸的**吸气阶段的预测压力（X）**和**实际压力（Y）**之间的平均绝对误差来计分。呼气阶段不计分。
- 推荐阅读 Kaggle 内的一篇 EDA（探索数据分析）来获取一些预备知识：[Ventilator Pressure Prediction: EDA, FE and models | Kaggle](https://www.kaggle.com/artgor/ventilator-pressure-prediction-eda-fe-and-models)



## 数据字段

官方数据页面 [Google Brain - Ventilator Pressure Prediction | Kaggle](https://www.kaggle.com/c/ventilator-pressure-prediction/data)

- `id` - globally-unique time step identifier across an entire file

  **全局唯一id**

- `breath_id` - globally-unique time step for breaths

  **呼吸id**

- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.

  **表示气道受限程度的肺部属性（单位：cmH2O/L/S）。从物理学上讲，这是每一个流量变化（每一个时间的空气量）的压力变化。直观地讲，我们可以想象通过吸管吹起一个气球。我们可以通过改变吸管的直径来改变R，R越高就越难吹。**

  **肺部阻塞情况，R越大，阻塞越大，呼吸越困难, pressure越高。**

- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.

  **肺部属性，表明肺部的顺应性如何（单位：毫升/厘米汞柱）。从物理上讲，这是每一个压力变化的体积变化。直观地说，我们可以想象同一个气球的例子。我们可以通过改变气球乳胶的厚度来改变C，C越高，乳胶越薄，越容易吹。**

  **肺部顺应性，C越大，呼吸越顺畅，pressure越低。**

- `time_step` - the actual time stamp.

  **时间戳**

- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.

  **吸气阀门打开值（0-100）**

- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.

  **呼气阀门打开值（0-1）**

- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

  **要预测的压力**




## 解决方案思路
本次竞赛的方案源自Kaggle大神 [@hengck23]([‎‎‎‎‎‎‎‎ | Grandmaster | Kaggle](https://www.kaggle.com/hengck23)) 在Discussion区发表的 [To learn or not to learn?](https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/278964) ，我们在此基础上做了适当修改后得到了top-5%的成绩，在此特别感谢 [@hengck23]([‎‎‎‎‎‎‎‎ | Grandmaster | Kaggle](https://www.kaggle.com/hengck23)) 的分享。最终方案大致如下：

- 特征工程使用的是 [Ventilator Train classification | Kaggle](https://www.kaggle.com/takamichitoda/ventilator-train-classification) 中的 Feature Engineering 部分

- 模型采用的是bidirectional lstm
- Loss函数 smoothed l1 loss
- 调参完毕后，使用全量数据训练（没有留验证集）
- 全量训练配合20个seed，得到20个输出结果，并且进行简单的加权平均



## 比赛上分历程

1. 使用 [@hengck23]([‎‎‎‎‎‎‎‎ | Grandmaster | Kaggle](https://www.kaggle.com/hengck23)) 提供的原始代码直接运行，得到一个单一的10fold切分的csv文件，Public LB : 0.1710
2. 原始代码跑一个10fold，10个csv结果进行简单的平均融合，Public LB : 0.1420
3. 改成全量训练数据，跑10个seed并简单平均融合，Public LB : 0.1380
4. [Ventilator Train classification | Kaggle](https://www.kaggle.com/takamichitoda/ventilator-train-classification) 中的 Feature Engineering 部分作为特征，并且跑一个单一的10fold切分的csv文件，Public LB : 0.1633
5. 使用新特征，并改成全量的训练数据，跑10个seed并简单平均融合，Public LB : 0.1322
6. 再对 batch_size、learning_rate、optimizer调参，10个seed并简单平均融合，Public LB : 0.1303
7. 20个seed，Public LB : 0.1292
8. 40个seed，并挑选20个相关系数小的csv文件，并简单平均融合，Public LB : 0.1287



## 特征工程

```python
#### 以下部分特征来自 https://www.kaggle.com/takamichitoda/ventilator-train-classification  
def add_feature(df):
    df['time_delta'] = df.groupby('breath_id')['time_step'].diff().fillna(0) # 时间差（或持续时间）
    df['delta'] = df['time_delta'] * df['u_in'] # 吸气阀门打开值*持续时间
    df['area'] = df.groupby('breath_id')['delta'].cumsum() # # 吸气阀门打开值*持续时间 的前向累计值

    df['cross']= df['u_in']*df['u_out'] # 吸气阀门打开值*呼气阀门打开值
    df['cross2']= df['time_step']*df['u_out'] # 呼气阀门打开值*持续时间
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum() # 每次呼吸中的 吸气阀门打开值 的总量
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum() # 每次呼吸中行为的总计数
    df['u_in_cummean'] =df['u_in_cumsum'] / df['count'] # 每次呼吸中的 吸气阀门打开值 的总量/每次呼吸中行为的总计数
    
    df = df.drop(['count','one'], axis=1)
    return df

def add_lag_feature(df):
    for lag in range(1, USE_LAG+1):
        df[f'breath_id_lag{lag}']=df['breath_id'].shift(lag).fillna(0) # 滑动lag行
        # 滑动lag行后，breathid仍然相同的行
        df[f'breath_id_lag{lag}same']=np.select([df[f'breath_id_lag{lag}']==df['breath_id']], [1], 0)

        # 吸气或呼气阀门打开值 的滑动相关值，具体需要体会代码，比较难用文字描述
        df[f'u_in_lag_{lag}'] = df['u_in'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']
        df[f'u_in_time{lag}'] = df['u_in'] - df[f'u_in_lag_{lag}']
        df[f'u_out_lag_{lag}'] = df['u_out'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']

    # breath_time
    df['time_step_lag'] = df['time_step'].shift(1).fillna(0) * df[f'breath_id_lag{lag}same'] # 滑动时间戳
    df['breath_time'] = df['time_step'] - df['time_step_lag'] # 原时间 和 滑动时间之差

    drop_columns = ['time_step_lag']
    drop_columns += [f'breath_id_lag{i}' for i in range(1, USE_LAG+1)]
    drop_columns += [f'breath_id_lag{i}same' for i in range(1, USE_LAG+1)]
    df = df.drop(drop_columns, axis=1)

    # fill na by zero
    df = df.fillna(0)
    return df

c_dic = {10: 0, 20: 1, 50:2} # C值的映射表
r_dic = {5: 0, 20: 1, 50:2} # R值的映射表
rc_sum_dic = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])} # C和R求和映射表
rc_dot_dic = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])} # C和R乘积映射表  

def add_category_features(df):
    # 将C和R值，分别映射成0,1,2
    df['C_cate'] = df['C'].map(c_dic)
    df['R_cate'] = df['R'].map(r_dic)
    df['RC_sum'] = (df['R'] + df['C']).map(rc_sum_dic) # C和R求和并映射
    df['RC_dot'] = (df['R'] * df['C']).map(rc_dot_dic) # C和R乘积并映射
    return df
    
df = add_feature(df)
df = add_lag_feature(df)
df = add_category_features(df)
```



## 模型代码

```python
class Net(nn.Module):
    def __init__(self, in_dim=10):
        super().__init__()
        # embedding层
        self.seq_emb = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
        )
        # 双向LSTM
        self.lstm = nn.LSTM(64, 256, batch_first=True, bidirectional=True, dropout=0.0, num_layers=4)

        # 输出头的层
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 256 * 2),
            nn.LayerNorm(256 * 2),
            nn.ReLU(),
            nn.Linear(256 * 2, 950),
        )

        self.pressure_in  = nn.Linear(950, 1) # 吸气阶段的压力
        self.pressure_out = nn.Linear(950, 1) # 呼气阶段的压力
        
        # LSTM的初始化
        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                print(f'init {m}')
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, x):
        batch_size = len(x)
        seq_x = x
        emb_x = self.seq_emb(seq_x)
        out, _ = self.lstm(emb_x, None) 
        logits = self.head(out)
        pressure_in  = self.pressure_in(logits).reshape(batch_size,80)
        pressure_out = self.pressure_out(logits).reshape(batch_size,80)
        return pressure_in, pressure_out
```



## 代码、数据集

+ 代码
  + dataset.py 数据集的构造
  + model.py 模型的构造
  + run_train_fold1.py 单次训练/验证的主代码
  + loop_run.py 用做连续跑20个seed结果
  + run_submit.py 生成可用于ensemble或者submit的csv文件
+ 数据集
  本次比赛没有使用外部数据集，请至kaggle下载官网数据即可
  [Google Brain - Ventilator Pressure Prediction | Kaggle](https://www.kaggle.com/c/ventilator-pressure-prediction/data)