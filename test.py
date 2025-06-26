import args.base

import os, time, logging, argparse, cv2, gc, torch, tqdm
import numpy as np

from utils.data_io import save_pfm, write_depth_img, save_camfile


@ torch.no_grad()
def test(cmd_args, test_args, test_batches):
    # creat model
    model = test_args.model

    # load breakpoint
    if cmd_args.pre_model is not None:
        checkpoint = torch.load(cmd_args.pre_model) # map_location=torch.device("cpu")
        model.load_state_dict(checkpoint["model"], strict=True)

    # load to device
    model.to(test_args.DEVICE)

    # test
    model.eval()
    progress_bar = tqdm.tqdm(desc="MVS test", total=len(test_batches))

    for iteration, data in enumerate(test_batches):

        torch.cuda.empty_cache()
        data_batch = {k: v.to(test_args.DEVICE) for k, v in data.items() if isinstance(v, torch.Tensor)}

        start_time = time.time()
        outputs = model(data_batch["imgs"], data_batch["extrinsics"], data_batch["intrinsics"], data_batch["depth_range"])

        # logging.info("batch: " + str(iteration + 1) + "/" + str(len(test_batches)) +
        #              " time: {:.3f}".format(time.time() - start_time) +
        #              " memory: {:.2f}".format(torch.cuda.max_memory_allocated() / (1024 ** 3)) + "GB")
        progress_bar.set_postfix({
            "forward time": "{:.3f}s".format(time.time() - start_time),
            "memory": "{:.2f}GB".format(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "output": os.path.join(test_args.output_path, data["filename"][0].format('', ''))
        })
        progress_bar.update(1)

        # save depth map, confidence map, depth img, imgs, cams
        for filename, depth, confidence, img, intrinsic, extrinsic, depth_range in \
                zip(data["filename"],
                    outputs["depth"],
                    outputs["confidence"],
                    data_batch["imgs"][0],
                    data_batch["intrinsics"][0], data_batch["extrinsics"][0], data_batch["depth_range"]):  # (B,H,W)
            depth_filename = os.path.join(test_args.output_path, filename.format('depth_est', '.pfm'))
            depthimg_filename = os.path.join(test_args.output_path, filename.format('depth_est', '.png'))
            conf_filename = os.path.join(test_args.output_path, filename.format('confidence', '.pfm'))
            confimg_filename = os.path.join(test_args.output_path, filename.format('confidence', '.png'))

            cam_filename = os.path.join(test_args.output_path, filename.format('cams', '_cam.txt'))
            img_filename = os.path.join(test_args.output_path, filename.format('images', '.jpg'))

            os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(conf_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)

            # save depth maps
            save_pfm(depth_filename, depth.cpu())
            write_depth_img(depthimg_filename, depth.cpu().numpy())
            # Save prob maps
            save_pfm(conf_filename, confidence.cpu())
            write_depth_img(confimg_filename, confidence.cpu().numpy())

            # save cams, img
            save_camfile(intrinsic.cpu().numpy(), extrinsic.cpu().numpy(), depth_range.cpu().numpy(), cam_filename)
            img = np.clip(np.transpose(img.cpu().numpy(), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_filename, img_bgr)

            # logging.info("save in: " + depth_filename)

    torch.cuda.empty_cache()
    progress_bar.close()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tanks check parameter setting')
    parser.add_argument('-p', '--pre_model', default=None, type=str, help='Pre training model')
    parser.add_argument('-d', '--dataset', default='dtu', type=str, choices=['dtu', 'tanks', 'eth3d', 'custom'], help='Set dataset')

    shell_args = parser.parse_args()
    logging.info(shell_args)

    if shell_args.dataset == "dtu":
        from args.dtu import Test
        from load.dtu import LoadDataset
    elif shell_args.dataset == "tanks":
        from args.tanks import Test
        from load.tanks import LoadDataset
    elif shell_args.dataset == "eth3d":
        from args.eth3d import Test
        from load.eth3d import LoadDataset
    elif shell_args.dataset == "custom":
        from args.custom import Test
        from load.custom import LoadDataset
    else:
        print("Error dataset")
        exit()

    test_args = Test()
    test_args.show_args()

    # load dataset
    test_dataset = LoadDataset(test_args)
    test_batches = torch.utils.data.DataLoader(test_dataset,
                              batch_size=test_args.batch_size,
                              num_workers=test_args.nworks,
                              shuffle=False, pin_memory=True, drop_last=False)

    from args.net import model
    test_args.model = model

    test(shell_args, test_args, test_batches)

