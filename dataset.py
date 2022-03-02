from common import *
from ventilator import *

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html


#---
data_dir = root_dir+'/ventilator-pressure-prediction'


i_col=[
    'id',
    'breath_id',
]
# x_col=[
#     'time_step',
#     'u_in',
#     'u_out',
#     'R',
#     'C',
# ]
y_col=[
    'pressure',
]


USE_LAG = 4
CONT_FEATURES = ['u_in', 'u_out', 'time_step'] + ['u_in_cumsum', 'u_in_cummean', 'area', 'cross', 'cross2'] + ['R_cate', 'C_cate']
LAG_FEATURES = ['breath_time']
LAG_FEATURES += [f'u_in_lag_{i}' for i in range(1, USE_LAG+1)]
LAG_FEATURES += [f'u_in_time{i}' for i in range(1, USE_LAG+1)]
LAG_FEATURES += [f'u_out_lag_{i}' for i in range(1, USE_LAG+1)]
x_col = CONT_FEATURES + LAG_FEATURES


def make_df(mode='train'):
    if mode=='train':
        df = pd.read_csv(data_dir + '/train.csv')
    if mode=='test':
        df = pd.read_csv(data_dir + '/test.csv')
        df.loc[:,'pressure']=-1

    #### 以下部分特征来自 https://www.kaggle.com/ventilator-train-classification
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
    ####


    #---
    print('df.shape', df.shape)
    print(df.columns)
    return df


def make_fold(df, mode='train-1', ):
    if 'train' in mode:
        # # 进行交叉验证的kfold切分数据
        # fold = int(mode[-1])
        # kf = KFold(n_splits=1000, random_state=123, shuffle=True)
        # train_idx, valid_idx = [],[]
        # for t, v in kf.split(df['breath_id'][::80]):
        #     train_idx.append(t)
        #     valid_idx.append(v)
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # print(len(train_idx), len(valid_idx), len(df['breath_id'][::80]))
        # print(len(train_idx[1]), len(valid_idx[1]), len(df['breath_id'][::80]))
        # return train_idx[fold], valid_idx[fold]
        
        return list(range(len(df['breath_id'][::80]))), [] # 如果全量数据训练，直接输出整个df

    if 'test' in mode:
        valid_idx = np.arange(len(df)//80)
        return valid_idx



class VentilatorDataset(Dataset):
    def __init__(self, df, idx, scaler):
        super().__init__()
        self.length = len(idx)
        self.idx = idx

        feature = scaler.transform(df[x_col].values).astype(np.float32)
        self.feature   = feature.reshape(-1, 80, len(x_col))
        self.pressure  = df[y_col].values.astype(np.float32).reshape(-1,80)
        self.u_out     = df['u_out'].values.astype(np.float32).reshape(-1,80)
        self.breath_id = df['breath_id'].values[::80]
        zz=0


    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        i = self.idx[index]
        r = {
            'index'     : index,
            'breath_id' : self.breath_id[i],
            'u_out'     : self.u_out[i],
            'pressure'  : self.pressure[i],
            'feature'   : self.feature[i],
        }
        return r



#########################################################


def run_check_dataset():
    # 测试本文件的代码是否能够正常运行
    df = make_df(mode='train')
    print('df', df.shape)
    #---
    scaler = RobustScaler()
    scaler.fit_transform(df[x_col])
    #---
    train_idx, _ = make_fold(df, mode='train-1')
    dataset = VentilatorDataset(df, train_idx, scaler)
    print(dataset)

    for i in range(5):
        #i = np.random.choice(len(dataset))#272 #
        r = dataset[i]

        print('---')
        print('index     :', r['index'])
        print('breath_id    :', r['breath_id'])
        print('pressure  :', r['pressure'].shape)
        print('u_out     :', r['u_out'].shape)
        print('feature   :', r['feature'].shape)

        if 0:
            plt.clf()
            plt.plot(r['feature' ], c='gray')
            plt.plot(r['pressure'], label='pressure', c='red')
            plt.legend()
            plt.waitforbuttonpress()
            #plt.show()

        if 1:
            j = train_idx[i]
            d = df.iloc[j*80:(j+1)*80]
            d0 = df.iloc[j*80]

            print('breath_id  :', d0['breath_id'])
            print('R :', d0['R'])
            print('C :', d0['C'])

            #---
            plt.clf()
            plt.plot(d['time_step'],d['pressure'], label='pressure',c='red')
            plt.plot(d['time_step'],d['u_in'], label='u_in',c='green')
            plt.plot(d['time_step'],d['u_out'], label='u_out',c='blue')

            # p = d['pressure'].values
            # u_in = d['u_in'].values
            # plt.plot(p[1:]-p[:-1], label='pressure')
            # plt.plot(u_in[1:]-u_in[:-1], label='u_in')

            plt.legend()
            plt.waitforbuttonpress(1)
            #plt.show()


    loader = DataLoader(
        dataset,
        sampler = RandomSampler(dataset),
        batch_size  = 8,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = True,
    )
    for t,batch in enumerate(loader):
        if t>30: break

        print(t, '-----------')
        print('index : ', batch['index'])
        print('breath_id : ', batch['breath_id'])
        print('u_out : ')
        print('\t', batch['u_out'].shape, batch['u_out'].is_contiguous())
        print('pressure : ')
        print('\t', batch['pressure'].shape, batch['pressure'].is_contiguous())
        print('feature : ')
        print('\t', batch['feature'].shape, batch['feature'].is_contiguous())
        print('')


##################################################################################
if __name__ == '__main__':
    run_check_dataset()

