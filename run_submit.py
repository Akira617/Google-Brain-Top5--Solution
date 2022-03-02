import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from common import *
from model import *
from dataset import *

matplotlib.use('TkAgg')
#----------------
import torch.cuda.amp as amp
is_amp = True  #True #False


#----------------



# start here !


def run_submit(fold, iter_num):
    # fold = 52
    # iter_num = 140000
    out_dir = root_dir + f'/hengck23/output/takamichitoda-18/seed{fold}'
    initial_checkpoint = \
        out_dir + f'/checkpoint/{iter_num}.model.pth'

    #augment = ['none','shift1','shift2','shift3','shift4'],
    #mode='local'   # remote
    mode='remote'   # remote

    # setup logging, etc ---------
    model_name = initial_checkpoint.split('/')[-1][:-4]
    submit_dir = out_dir + '/valid/submit_%s_%s' % (mode,model_name)
    os.makedirs(submit_dir, exist_ok=True)
    print(submit_dir)

    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')

    #net
    net = Net(len(x_col))
    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)['state_dict']
    net.load_state_dict(state_dict, strict=True)  # True
    net = net.cuda()
    net = net.eval()

    #dataset
    if mode=='local':
        df = make_df(mode='train')
        train_idx, valid_idx = make_fold(df, mode='train-%d'%fold)
        #valid_idx = train_idx

    if mode=='remote':
        df = make_df(mode='test')
        valid_idx = make_fold(df, mode='test')

    #---
    id =  np.concatenate([
        df.iloc[i*80:(i+1)*80]['id'] for i in valid_idx
    ])

    scaler = RobustScaler()
    scaler.fit_transform(df[x_col])

    valid_dataset = VentilatorDataset(df, valid_idx, scaler)
    valid_loader  = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = 128,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = True,
        #collate_fn  = null_collate,
    )
    log.write('fold  : %d\n'%(fold))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')

    #start here!!!! 
    if 1:
        valid_pressure = []
        valid_u_out = []
        valid_truth = []
        valid_num = 0

        net.eval()
        start_timer = timer()
        for t, batch in enumerate(valid_loader):
            batch_size = len(batch['index'])
            pressure_truth = batch['pressure'].cuda()
            u_out = batch['u_out'].cuda()
            x = batch['feature'].cuda()

            with torch.no_grad():
                with amp.autocast(enabled=is_amp):
                    pressure_in, pressure_out = data_parallel(net, (x))
                    pressure = pressure_in*(1-u_out) + pressure_out*u_out

                valid_num += batch_size
                valid_pressure.append(pressure.data.cpu().numpy())
                valid_truth.append(pressure_truth.data.cpu().numpy())
                valid_u_out.append(u_out.data.cpu().numpy())
                print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)

        assert(valid_num == len(valid_loader.dataset))
        print('')

        #save
        pressure = np.concatenate(valid_pressure)
        truth = np.concatenate(valid_truth)
        u_out = np.concatenate(valid_u_out)

        log.write('pressure  : %s\n' % str(pressure.shape))
        log.write('u_out  : %s\n' % str(u_out.shape))


        log.write('submit_dir  : %s\n' % submit_dir)
        submit_df = pd.DataFrame({
            'id': id,
            'pressure': pressure.reshape(-1),
        })
        submit_df.to_csv(submit_dir+f'/{out_dir.split("/")[-2].split("-")[-1]}_{fold}_{model_name.split(".")[0]}_df_submit.csv', index=False)
        np.save(submit_dir+f'/{out_dir.split("/")[-2].split("-")[-1]}_{fold}_{model_name.split(".")[0]}_valid_idx.npy', valid_idx)

        if mode=='local':
            mask = 1 - u_out
            mae = np.abs(mask * (truth - pressure))
            mae = np.sum(mae) / np.sum(mask)

            log.write('\tinitial_checkpoint  = %s\n' % initial_checkpoint)
            log.write('\tfold = %d\n' % fold)
            log.write('\t**ALL**\n')
            log.write('\t\tmae  = %f\n' % mae)
            log.write('\n')


def run_show_results():
    df = make_df(mode='train')
    submit_dir = root_dir + '/hengck23/result/lstm-hengck23-split2-32-f8/fold1/valid/submit_local_00100000.model'

    valid_idx = np.load(submit_dir+'/valid_idx.npy')
    submit_df = pd.read_csv(submit_dir+'/df_submit.csv')

    num_valid = len(valid_idx)

    pressure = submit_df.pressure.values.reshape(num_valid,80)
    zz=0
    loss = []
    for b in range(num_valid):
        i = valid_idx[b]
        d = df.iloc[i*80:(i+1)*80]
        u_out = d['u_out']
        p_hat = d['pressure']
        p = pressure[b]
        l = ((1 - u_out) * (np.abs(p - p_hat))).sum() / (1 - u_out).sum()
        loss.append(l)
        print('\r',b,l,end='')
    loss = np.array(loss)
    print(loss.shape)
    print(loss.mean())

    argsort = np.argsort(-loss)
    #argsort = np.argsort(loss)
    for a in range(1000):
        b = argsort[a]
        i = valid_idx[b]
        d = df.iloc[i*80:(i+1)*80]

        u_out = d['u_out'].values
        p_hat = d['pressure'].values
        t = d['time_step'].values
        R = int(d['R'].values[0])
        C = int(d['C'].values[0])
        u_in = d['u_in'].values
        l = loss[b]
        p = pressure[b]
        if (R==50) and (C==50):

            plt.clf()
            plt.plot(t, p, label='p')
            plt.scatter(t, p, c='k')
            plt.plot(t, p_hat, label='p_hat')
            plt.plot(t, u_in, label='u_in')
            plt.plot(t, u_out, label='u_out')
            plt.legend()
            plt.ylim([-5, 65])
            plt.title('C=%d, R=%d : loss=%0.5f' % (C, R, l))
            plt.waitforbuttonpress()


# main
if __name__ == '__main__':
    for fold in [101,111,121,131]:
        for iter_num in ["00140000", "00160000"]:
            run_submit(fold, iter_num)
    # run_show_results()
