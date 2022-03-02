from common import *
from einops.layers.torch import Rearrange


#----------------------------------------------------------------------------------

def mask_huber_loss(predict, truth, m, delta=0.1):
    loss = F.huber_loss(predict[m], truth[m], delta=delta)
    return loss

def mask_l1_loss(predict, truth, m):
    loss = F.l1_loss(predict[m], truth[m])
    return loss

# 最终方案用的就是这个loss
def mask_smooth_l1_loss(predict, truth, m, beta=0.1):
    loss = F.smooth_l1_loss(predict[m], truth[m], beta=beta)
    return loss


# 构建的模型
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



def run_check_net(): 
    # 测试本文件的代码是否能够正常运行
    batch_size = 10
    length = 80
    in_dim = 10
    x  = torch.randn((batch_size, length, in_dim-2))
    rc = torch.from_numpy(np.concatenate([
        np.random.choice([ 5, 20, 50],(batch_size, length, 1)),
        np.random.choice([10, 20, 50],(batch_size, length, 1)),
    ],2)).float()
    x  = torch.cat([x,rc], 2)

    net = Net(in_dim)
    pressure_in, pressure_out = net(x)
    print('x  :', x.shape)
    print('pressure_in  :', pressure_in.shape)
    print('pressure_out :', pressure_out.shape)

##################################################################################
if __name__ == '__main__':
    run_check_net()