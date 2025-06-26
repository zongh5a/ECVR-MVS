import os, gc, tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import argparse, time, torch, math
import numpy as np
from plyfile import PlyData, PlyElement

import conf
from utils import read_pfm, save_pfm, read_pairfile, read_img, read_cam_file, save_mask, bilinear_sampler

@torch.no_grad()
def filter(dataset_root, scan, img_folder, cam_folder,
           eval_folder, filter_folder, outply_folder,
           photo_threshold, nconditions, s, dist_diff, rel_diff, thres_view):
    """
    # 1. filtered with photo consistency and geo consistency
    # 2. map to world location,and save to ply
    @param dataset_root: input folder:scene,include camera and photo
    @param scan:
    @param img_folder:
    @param cam_folder:
    @param eval_folder: include depth_est,photo consistency
    @param filter_folder: output folder:mask, depth_filter,
    """

    scan_location = os.path.join(dataset_root, scan)
    pair_path = os.path.join(scan_location, "pair.txt")

    eval_location = os.path.join(eval_folder, scan)

    filter_workspace = os.path.join(eval_location, filter_folder)
    os.makedirs(filter_workspace, exist_ok=True)

    vertexs, vertex_colors = [], []
    num_viewpoint, pairs = read_pairfile(pair_path)

    progress_bar = tqdm.tqdm(desc=scan, total=len(pairs))

    accumulate_points = 0
    for ref_view, src_views in pairs:
        start_time = time.time()

        cam_path = os.path.join(eval_location, cam_folder,'{:0>8}_cam.txt'.format(ref_view))
        refimg_path = os.path.join(eval_location, img_folder,'{:0>8}.jpg'.format(ref_view))

        ref_intrinsics, ref_extrinsics = read_cam_file(cam_path)
        ref_img = read_img(refimg_path)

        depth_est_path = os.path.join(eval_location, "depth_est",'{:0>8}'.format(ref_view) + '.pfm')
        confidence_path = os.path.join(eval_location, "confidence",'{:0>8}'.format(ref_view) + '.pfm')

        ref_depth_est, scale = read_pfm(depth_est_path)
        confidence, _ = read_pfm(confidence_path)

        # to cuda
        ref_depth_est = torch.from_numpy(ref_depth_est.copy()).float().cuda()
        confidence = torch.from_numpy(confidence.copy()).float().cuda()
        ref_intrinsics = torch.from_numpy(ref_intrinsics.copy()).float().cuda()
        ref_extrinsics = torch.from_numpy(ref_extrinsics.copy()).float().cuda()
        h, w = confidence.shape

        # compute the geometric mask
        avg_mask, all_srcview_depth_ests, dynamic_mask_sum = 0, [], []
        for index, src_view in enumerate(src_views):

            src_cam_path=os.path.join(eval_location, cam_folder, '{:0>8}_cam.txt'.format(src_view))
            src_depth_est=os.path.join(eval_location, "depth_est", '{:0>8}'.format(src_view) + '.pfm')

            src_intrinsics, src_extrinsics = read_cam_file(src_cam_path)
            src_depth_est = read_pfm(src_depth_est)[0]

            src_depth_est = torch.from_numpy(src_depth_est.copy()).float().cuda()
            src_intrinsics = torch.from_numpy(src_intrinsics.copy()).float().cuda()
            src_extrinsics = torch.from_numpy(src_extrinsics.copy()).float().cuda()

            # check geometric consistency
            dynamic_masks, mask, depth_reprojected = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                                 src_depth_est, src_intrinsics, src_extrinsics,
                                                                                 s, dist_diff, rel_diff)

            dynamic_masks = [m.float() for m in dynamic_masks]
            if index == 0:
                dynamic_mask_sum = dynamic_masks
            else:
                for i in range(11-s):
                    dynamic_mask_sum[i] += dynamic_masks[i]

            avg_mask += mask
            all_srcview_depth_ests.append(depth_reprojected)

        if thres_view is not None:
            geo_mask = (avg_mask>=thres_view).float()
        else:
            geo_mask = 0
        if len(dynamic_mask_sum) != 0:
            for i in range(s, 11):
                geo_mask += (dynamic_mask_sum[i - s] >= i)  # iter result save in geo_mask
        else:
            continue

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (avg_mask + 1)

        # compose filter
        geo_mask = geo_mask >= nconditions      # condition filter
        conf_mask = confidence >= photo_threshold   # confidence filter
        final_mask = conf_mask & geo_mask

        # to numpy
        depth_est_averaged = depth_est_averaged[0].cpu().numpy()
        geo_mask = geo_mask[0].cpu().numpy()
        conf_mask = conf_mask.cpu().numpy()
        final_mask = final_mask[0].cpu().numpy()
        ref_intrinsics = ref_intrinsics.cpu().numpy()
        ref_extrinsics = ref_extrinsics.cpu().numpy()
        ref_depth_est = ref_depth_est.cpu().numpy()

        accumulate_points += final_mask.sum()

        progress_bar.set_postfix({
            "ref-view{:0>3}".format(ref_view,):
                "conf/geo/final-mask:{}/{}/{}".format( conf_mask.sum(), geo_mask.sum(), final_mask.sum()),
            "time": "{:.3f}s".format(time.time() - start_time),
            "accumulate points": "{:.3f}M".format(accumulate_points/(1024*1024))
        })
        progress_bar.update(1)

        # save mask
        save_mask(os.path.join(filter_workspace, "conf_{:0>8}.png".format(ref_view)), conf_mask)
        save_mask(os.path.join(filter_workspace, "geo_{:0>8}_.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(filter_workspace, "final_{:0>8}.png".format(ref_view)), final_mask)

        save_pfm(os.path.join(filter_workspace, "{}".format(ref_view)+"_" + "depth_est.pfm"),
                                ref_depth_est * final_mask.astype(np.float32))

        #######################################################################################

        # 2.map to world location,and save to ply
        height, width = depth_est_averaged.shape[:2]
        valid_points = final_mask

        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        color = ref_img[:h, :w, :][valid_points]

        # pix to camera ,to world
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    progress_bar.close()
    gc.collect()
    torch.cuda.empty_cache()

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)

    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    if outply_folder is None:
        PlyData([el]).write(os.path.join(eval_location, scan+".ply"))
        print("saving the final model to", os.path.join(eval_location, scan + ".ply"))
    else:
        os.makedirs(outply_folder, exist_ok=True)
        PlyData([el]).write(os.path.join(outply_folder, scan+".ply"))
        print("saving the final model to", os.path.join(outply_folder, scan + ".ply"))

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref,
                                depth_src, intrinsics_src, extrinsics_src,
                                s=2, thre1=4, thre2=1300.):
    height, width = depth_ref.shape
    batch = 1
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device),
                                  torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch, 1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch, 1, 1)
    inputs = [depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src]
    outputs = reproject_with_depth(*inputs)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = outputs
    # check |p_reproj-p_1|
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    masks = []
    for i in range(s, 11):
        if conf.dataset == "dtu":
            mask = torch.logical_and(dist < i / dist_diff, relative_depth_diff < math.log(max(i, 1.05), 10) / rel_diff)
        elif conf.dataset.startswith("tanks"):
            mask = torch.logical_and(dist < i / dist_diff, relative_depth_diff < i / rel_diff)
        else:
            mask = torch.logical_and(dist < i / dist_diff, relative_depth_diff < i / rel_diff)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    height, width = depth_ref.shape
    batch = 1
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    x_ref, y_ref = x_ref.reshape(batch, -1), y_ref.reshape(batch, -1)
    # reference 3D space

    A = torch.inverse(intrinsics_ref)
    B = torch.stack((x_ref, y_ref, torch.ones_like(x_ref).to(x_ref.device)), dim=1) * depth_ref.reshape(batch, 1, -1)
    xyz_ref = torch.matmul(A, B)

    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                        torch.cat((xyz_ref, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].reshape([batch, height, width]).float()
    y_src = xy_src[:, 1].reshape([batch, height, width]).float()

    # print(x_src, y_src)
    sampled_depth_src = bilinear_sampler(depth_src.view(batch, 1, height, width), torch.stack((x_src, y_src), dim=-1).view(batch, height, width, 2))

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                        torch.cat((xy_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1) * sampled_depth_src.reshape(batch, 1, -1))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.inverse(extrinsics_src)),
                                torch.cat((xyz_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].reshape([batch, height, width]).float()
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
    x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='filter to maks cloudpoints...')
    parser.add_argument('--clean', action='store_true', help='remove the output dir of scene')

    args = parser.parse_args()
    print(args)

    dataset_root, test_folder, img_folder, cam_folder, filter_folder, outply_folder = \
        conf.dataset_root, conf.test_folder, conf.img_folder, conf.cam_folder, conf.filter_folder, conf.outply_folder
    scenes, filter_paras = conf.scenes, conf.filter_paras

    # filter with photometric confidence maps and geometric constraints,then convert to world position
    for scene in scenes:
        start_time = time.time()

        nconditions, conf_thresh, s, dist_diff, rel_diff, thres_view = filter_paras[scene]
        filter(dataset_root, scene, img_folder, cam_folder,
               test_folder, filter_folder, outply_folder,
               photo_threshold=conf_thresh, nconditions=nconditions, s=s, dist_diff=dist_diff, rel_diff=rel_diff, thres_view=thres_view)

        print("all time: {:.3f}min".format((time.time()-start_time)/60.0))

        # To save disk space, del sence dir
        if args.clean:
            if os.path.exists(os.path.join(args.outply_folder, scene+".ply")):
                cmd = "rm -r " + os.path.join(args.eval_folder, scene)
                print(cmd)
                os.system(cmd)
            else:
                print("remove error!")




