import os

# dataset = "dtu"
# dataset = "tanks-inter"
# dataset = "tanks-adv"
# dataset = "eth3d"
dataset = "custom"

# input
dataset_root = "/hy-tmp"    # for finding pair.txt
test_folder = "/hy-tmp/outputs"
img_folder = "images"
cam_folder = "cams"

# output
filter_folder = "dypcd"
outply_folder = "/hy-tmp/outputs/plys"

if dataset == "dtu":
    dataset_root = os.path.join(dataset_root, "dtu1600x1200")
    labels = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
    scenes = ["scan" + str(label) for label in labels]

    # nconditions, conf, s, dist_diff, rel_diff, thres_view
    filter_paras = {scan: [1, 0.10, 1, 4, 1500, 4] for scan in scenes}           # acc.: 0.339046, comp.: 0.245294, overall: 0.292170

if dataset == "tanks-inter":
    dataset_root = os.path.join(dataset_root, "TanksandTemples", "intermediate", )
    scenes = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']

    # nconditions, conf, s, dist_diff, rel_diff, thres_view
    filter_paras = {
        'Family':       [1, 0.50, 2, 8, 1800, 7],
        'Francis':      [1, 0.60, 4, 4, 2000, 7],
        'Horse':        [1, 0.20, 1, 2, 1500, 7],
        'Lighthouse':   [1, 0.60, 2, 4, 1800, 7],
        'M60':          [1, 0.50, 2, 4, 1800, 7],
        'Panther':      [1, 0.40, 2, 4, 1800, 7],
        'Playground':   [1, 0.65, 2, 4, 1800, 7],
        'Train':        [1, 0.45, 2, 4, 1800, 7],
    }

if dataset == "tanks-adv":
    dataset_root = os.path.join(dataset_root, "TanksandTemples", "advanced")
    scenes = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']

    # nconditions, conf, s, dist_diff, rel_diff, thres_view
    filter_paras = {
                'Auditorium':   [1, 0.15, 1, 2, 600, 4],
                'Ballroom':     [1, 0.20, 2, 4, 1600, 7],
                'Courtroom':    [1, 0.20, 1, 3, 1500, 7],
                'Museum':       [1, 0.40, 1, 4, 2000, 7],
                'Palace':       [1, 0.30, 1, 4, 2000, 7],
                'Temple':       [1, 0.30, 1, 4, 1000, 7],
    }

if dataset == "eth3d":
    dataset_root = os.path.join(dataset_root, "eth3d")
    scenes = ["botanical_garden", "boulders", "bridge", "courtyard", "delivery_area",
             "door", "electro", "exhibition_hall", "facade", "kicker", "lecture_room",
             "living_room", "lounge", "meadow", "observatory", "office", "old_computer",
             "pipes", "playground", "relief", "relief_2", "statue", "terrace", "terrace_2", "terrains"]

    # nconditions, conf, s, dist_diff, rel_diff, thres_view
    filter_paras = {scan: [1, 0.10, 1, 4, 1500, 4] for scan in scenes}  # 1500

if dataset == "custom":
    dataset_root = r"E:\3DGS\dataset\scene2"    # for finding pair.txt
    scenes = ["scene2"]

    # nconditions, conf, s, dist_diff, rel_diff, thres_view
    filter_paras = {scan: [1, 0.30, 2, 4, 1800, 7] for scan in scenes}