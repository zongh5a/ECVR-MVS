import os


def get_img_path(dataset_path, scan_folder, view_id, lighting=None, dataset=''):
    if dataset in ["dtutrain", "dtuval"]:
        img_path = os.path.join(dataset_path, "Rectified", scan_folder+"_train","rect_{:0>3}_{}_r5000.png".format(view_id + 1, lighting))
    elif dataset in ["dtutest", "tanks", "eth3d"]:
        img_path = os.path.join(dataset_path, scan_folder, "images", '{:0>8}.jpg'.format(view_id))
    elif dataset in ["bldtrain", "bldval"]:
        img_path = os.path.join(dataset_path, '{}/blended_images/{:0>8}.jpg'.format(scan_folder, view_id))
    elif dataset in ["custom"]:
        img_path = os.path.join(dataset_path, scan_folder, "images_post", '{:0>8}.jpg'.format(view_id))
    else:
        img_path = None
        print("Unrecognized dataset!")

    return img_path


def get_cam_path(dataset_path, scan_folder, view_id, dataset):
    if dataset in ["dtutrain", "dtuval"]:
        cam_path = os.path.join(dataset_path, 'Cameras', '{:0>8}_cam.txt'.format(view_id))
    elif dataset == "dtutest":
        cam_path = os.path.join(dataset_path, scan_folder, "cams", '{:0>8}_cam.txt'.format(view_id))
    elif dataset == "tanks":
        cam_path = os.path.join(dataset_path, scan_folder, "cams", '{:0>8}_cam.txt'.format(view_id))    # _1
    elif dataset in ["bldtrain", "bldval"]:
        cam_path = os.path.join(dataset_path, '{}/cams/{:0>8}_cam.txt'.format(scan_folder, view_id))
    elif dataset == "eth3d":
        cam_path = os.path.join(dataset_path, scan_folder, "cams_1", '{:0>8}_cam.txt'.format(view_id))
    elif dataset in ["custom"]:
        cam_path = os.path.join(dataset_path, scan_folder, "cams", '{:0>8}_cam.txt'.format(view_id))
    else:
        cam_path = None
        print("Unrecognized dataset!")

    return cam_path


def get_depth_path(dataset_path, scan_folder, view_id, dataset):
    if dataset in ["dtutrain", "dtuval"]:
        depth_path = os.path.join(dataset_path, "Depths", scan_folder+"_train",'depth_map_{:0>4}.pfm'.format(view_id))
    elif dataset in ["bldtrain", "bldval"]:
        depth_path = os.path.join(dataset_path, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan_folder, view_id))
    else:
        depth_path = None
        print("Unrecognized dataset!")

    return depth_path


def get_confidenct_path(mode):
    if mode == "check":
        pass
    else:
        pass  # custom
