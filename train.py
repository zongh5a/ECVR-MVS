import args.base

import torch, argparse, os, logging, time
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.data_io import tocuda
from utils.tfblogs import save_tf_logs_train, save_tf_logs_val
from utils.scheduler import WarmupMultiStepLR



def train(shell_args, train_args, train_batches, val_args, val_batches, logger=None):
    # creat model, optimizer
    model = train_args.model
    optimizer = optim.Adam(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()),  'initial_lr': train_args.lr,}],
                lr=train_args.lr,
    )    # model.parameters()

    # load breakpoint
    start_epoch = train_args.start_epoch
    if shell_args.pre_model is not None:
        checkpoint = torch.load(shell_args.pre_model, )   # map_location=torch.device("cpu")
        model.load_state_dict(checkpoint["model"])
        if train_args.fine_tune:
            start_epoch = 1
            optimizer.param_groups[0]['lr'] = train_args.lr
        else:
            start_epoch = checkpoint["epoch"] + 1
            # optimizer.tensor.to GPU
            # optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    # scheduler
    milestones = [len(train_batches) * int(epoch_idx) for epoch_idx in train_args.lrepochs]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=train_args.lr_gamma, warmup_factor=1.0 / 3, warmup_iters=500,
                                     last_epoch=len(train_batches) * (start_epoch-1) - 1)   # len(train_batches) * start_epoch - 1

    # load to device
    if train_args.parallel:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(train_args.DEVICE)

    # train
    best_thresh5t_error = 1e2
    for epoch in range(start_epoch, train_args.max_epoch+1):
        epoch_loss, start_time, epoch_start_time = 0.0, time.time(), time.time()
        for batch, data_batch in enumerate(train_batches):
            model.train()
            data_batch = tocuda(data_batch, train_args.DEVICE, train_args.parallel)
            outputs = model(data_batch["imgs"], data_batch["extrinsics"], data_batch["intrinsics"], data_batch["depth_range"])
            loss, mask = train_args.model_loss(outputs, data_batch["ref_depths"], data_batch["depth_range"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            current_loss = loss.detach().item()
            epoch_loss += current_loss
            print("\r"+"Epoch: "+str(epoch)+
                  " batch: "+str(batch + 1)+ "/"+ str(len(train_batches))+
                  " lr: {:.6f}".format(optimizer.param_groups[0]['lr'])+
                  " time: {:.3f}".format(time.time() - start_time)+
                  " loss:{: .3f}".format(current_loss) +
                  " memory: {:.3f}GB".format(torch.cuda.max_memory_allocated()/(1024**3)) +
                  " epoch res time： {:.3f}min".format((time.time() - start_time)*(len(train_batches)-batch-1)/60),
                  end="", flush=True)
            start_time = time.time()

            # tf log
            global_step = len(train_batches) * (epoch - 1) + batch
            if global_step % shell_args.freq == 0:
                save_tf_logs_train(logger, data_batch, outputs, mask, loss, global_step)

            # val
            if train_args.val and (global_step % (shell_args.freq*40) == 100):
                depth_prob, prob_conf, thresh5t_error = val(model, val_args, val_batches, epoch)
                save_tf_logs_val(logger, depth_prob, prob_conf, thresh5t_error, global_step)

                if best_thresh5t_error > thresh5t_error:
                    best_thresh5t_error = thresh5t_error
                    _save_paras(epoch, model, optimizer, shell_args, global_step)

        logging.info("\n Epoch: " + str(epoch) +" loss:" + str(epoch_loss / len(train_batches)) +
                     " time:{: .3f}min".format((time.time() - epoch_start_time)/60))

        # save epoch loss
        with open(os.path.join(train_args.pth_path, "epoch_loss.txt"), "a") as f:
            f.write(str(epoch_loss / len(train_batches)) + "\n")

        # save epoch model
        if epoch % 1 == 0:
            _save_paras(epoch, model, optimizer, shell_args)


def val(model, val_args, val_batches, epoch):
    model.eval()
    with torch.no_grad():
        thresh5t_error = 0.0
        start_time = time.time()
        for batch, data_batch in enumerate(val_batches):
            data_batch = tocuda(data_batch, val_args.DEVICE, val_args.parallel)
            outputs = model(data_batch["imgs"], data_batch["extrinsics"], data_batch["intrinsics"], data_batch["depth_range"])

            depth_min, depth_max = \
                data_batch["depth_range"][:, 0].view(-1, 1, 1), \
                data_batch["depth_range"][:, 1].view(-1, 1, 1)
            depth_prob = outputs["depth"]
            depth_gt = list(data_batch["ref_depths"].values())[-1]
            mask = depth_gt > depth_min
            thresh_5t = 5e-3 * (depth_max - depth_min)  # the thresh is 5/1000 for depth range
            thresh5t_error += torch.mean(
                (torch.abs(depth_prob[mask] - depth_gt[mask]) > thresh_5t).float()).detach().item()  # !!!:  <

            print("\r" + "Val Epoch: " + str(epoch) +
                  " batch: " + str(batch + 1) + "/" + str(len(val_batches)) +
                  " time: {:.3f}".format(time.time() - start_time) +
                  " memory: {:.3f}MB".format(torch.cuda.max_memory_allocated() / (1024 ** 2)) +
                  " epoch res time： {:.3f}min".format(
                      (time.time() - start_time) * (len(val_batches) - batch - 1) / 60),
                  end="", flush=True)
            start_time = time.time()

    return depth_prob, outputs["confidence"], thresh5t_error / (batch + 1)


def _save_paras(epoch, model, optimizer, cmd_args, global_step=None):
    checkpoint = { 'epoch': epoch, 'model': None, 'optimizer': optimizer.state_dict()}
    checkpoint['model'] = model.module.state_dict() if train_args.parallel else model.state_dict()

    if global_step is None:
        save_path = os.path.join(train_args.pth_path, cmd_args.dataset + "_" + str(epoch) + ".pth")
    else:
        save_path = os.path.join(train_args.pth_path, cmd_args.dataset + "_" + str(epoch) + "_" + str(global_step) + ".pth")

    torch.save(checkpoint, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DTU train parameter setting')
    parser.add_argument('-p', '--pre_model', default=None, type=str, help='Pre training model or last model')
    parser.add_argument('-d', '--dataset', default='dtu', type=str, choices=['dtu', 'bld'], help='Set dataset')
    parser.add_argument('--freq', default=50, type=int, help='save tensorboard freq')
    parser.add_argument('-l', '--shell_label', default="", type=str, help='show train condition in ps -aux')

    shell_args = parser.parse_args()
    logging.info(shell_args)

    # train_args, val_args
    if shell_args.dataset == "dtu":
        import args.dtu as pre_args
        from load.dtu import LoadDataset
        from mvs.loss import dtu_loss as model_loss
    elif shell_args.dataset == "bld":
        import args.bld as pre_args
        from load.bld import LoadDataset
        import mvs.loss.bld_loss as model_loss
    else:
        print("Error dataset")
        exit()

    train_args = pre_args.Train()
    train_args.show_args()
    val_args = None if not train_args.val else pre_args.Val(); pre_args.Val().show_args()

    # model, loss
    from args import net
    train_args.model = args.net.model
    train_args.model_loss = model_loss

    # load dataset
    train_dataset = LoadDataset(train_args)
    train_batches = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True,
                               num_workers=train_args.nworks,
                               drop_last=True, pin_memory=True, )
    val_batches = None
    if train_args.val:
        val_dataset = LoadDataset(val_args)
        val_batches = DataLoader(val_dataset, batch_size=val_args.batch_size, shuffle=True, # False
                                   num_workers=val_args.nworks,
                                   drop_last=True, pin_memory=True, )

    # tensorboard
    from tensorboardX import SummaryWriter
    logger = SummaryWriter(train_args.pth_path)


    train(shell_args, train_args, train_batches, val_args, val_batches, logger)

    logger.close()