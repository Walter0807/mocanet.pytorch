import sys, os
sys.path.insert(0, os.getcwd())
import json
import argparse
import numpy as np
from itertools import combinations
from scripts.visualize.visualize_keypoints_npy import render_preview
from lib.util.motion import rotate_basis, get_limb_lengths
import imageio


def load_and_preprocess(path):
    motion3d = np.load(path)
    _, _, T = motion3d.shape
    T = (T // 8) * 8
    motion3d = motion3d[:, :, :T]
    motion3d[1, :, :] = (motion3d[2, :, :] + motion3d[5, :, :]) / 2
    motion3d[8, :, :] = (motion3d[9, :, :] + motion3d[12, :, :]) / 2
    motion3d[:, 0, :] = - motion3d[:, 0, :]
    motion3d[:, 2, :] = - motion3d[:, 2, :]
    motion2d = motion3d[:, [0, 2], :]
    return motion3d, motion2d


def rotate_and_maybe_project(motion3d, angles, project_2d=False):

    local3d = rotate_basis(np.eye(3), angles)
    motion3d = local3d @ motion3d

    if project_2d:
        return motion3d[:, [0, 2], :]
    else:
        return motion3d


def relocate(motion, fix_hip):
    if fix_hip:
        motion = motion - motion[8:9, :, :]
    else:
        center = motion[8, :, 0]
        motion = motion - center[np.newaxis, :, np.newaxis]
    return motion


def get_height(motion3d):

    limb_lengths = get_limb_lengths(motion3d)
    height = limb_lengths[0] + limb_lengths[4] + limb_lengths[6] + limb_lengths[7]
    return height


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data/mixamo/36_800_24/test_random_rotate")
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=False)
    parser.add_argument('--fix_hip', action="store_true")
    parser.add_argument('--norm_height', action="store_true")
    parser.add_argument('--height_space', action="store_true")
    parser.add_argument('--screen_size', type=int, default=512)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--grouping', type=str, default="character", choices=["character", "none"])
    args = parser.parse_args()

    description = json.load(open("data/mixamo/36_800_24/mse_description.json"))
    chars = list(description.keys())

    char_heihgts = []
    for i, char in enumerate(chars):
        motions = description[char]
        seqs = []
        for mot in motions:
            motion_path = os.path.join(args.data_dir, char, mot, "{}.npy".format(mot))
            motion3d, _ = load_and_preprocess(motion_path)
            seqs.append(motion3d)
        seqs = np.concatenate(seqs, axis=-1)
        char_heihgts.append(get_height(seqs))
    char_heihgts = np.array(char_heihgts)
    if not args.height_space:
        char_heihgts = char_heihgts / np.mean(char_heihgts)
    char_heihgts = { char:char_heihgts[i] for i, char in enumerate(chars) }
    print(char_heihgts)

    results = {}
    cnt = 0

    for char1, char2 in combinations(chars, 2):

        motions1 = description[char1]
        motions2 = description[char2]

        for i, mot1 in enumerate(motions1):
            for j, mot2 in enumerate(motions2):

                gt_path_ab = os.path.join(args.data_dir, char2, mot1, "{}.npy".format(mot1))
                gt_path_ba = os.path.join(args.data_dir, char1, mot2, "{}.npy".format(mot2))

                path_ab = os.path.join(args.in_dir, "motion_{}_{}_body_{}_{}_3d.npy".format(char1, i, char2, j))
                path_ba = os.path.join(args.in_dir, "motion_{}_{}_body_{}_{}_3d.npy".format(char2, j, char1, i))

                GT_ab, gt_ab = load_and_preprocess(gt_path_ab)
                GT_ba, gt_ba = load_and_preprocess(gt_path_ba)

                GT_ab = relocate(GT_ab, args.fix_hip)
                GT_ba = relocate(GT_ba, args.fix_hip)

                infered_ab = np.load(path_ab)
                infered_ba = np.load(path_ba)
                infered_ab = relocate(infered_ab, args.fix_hip)
                infered_ba = relocate(infered_ba, args.fix_hip)

                diff_ab = GT_ab - infered_ab
                diff_ba = GT_ba - infered_ba
                diff_ab, diff_ba = diff_ab * (args.screen_size / 4), diff_ba * (args.screen_size / 4)

                if args.norm_height:
                    denom_a = char_heihgts[char1] * ((args.screen_size / 4) if args.height_space else 1)
                    denom_b = char_heihgts[char2] * ((args.screen_size / 4) if args.height_space else 1)
                    diff_ab = diff_ab / denom_b
                    diff_ba = diff_ba / denom_a

                results["motion_{}_{}_body_{}_{}".format(char1, i, char2, j)] = diff_ab
                results["motion_{}_{}_body_{}_{}".format(char2, j, char1, i)] = diff_ba

                if args.render and cnt % 432 == 7:
                    video_gt_ab = render_preview(GT_ab[:, [0,2], :], h=512, w=512, scale=128, color='#a50b69#b73b87#db9dc3',
                                                 transparency=False, disable_smooth=True)
                    video_infered_ab = render_preview(infered_ab[:, [0,2], :], h=512, w=512, scale=128, color='#4076e0#40a7e0#40d7e0',
                                                 transparency=False, disable_smooth=True)
                    video_ab = 0.5 * video_gt_ab + 0.5 * video_infered_ab
                    video_ab = video_ab.astype(np.uint8)
                    seq_len, _, _, _ = video_ab.shape
                    imageio.mimwrite("motion_{}_{}_body_{}_{}_3d.mp4".format(char1, i, char2, j), video_ab, format="mp4", fps=25)

                cnt += 1


    if args.grouping == "character":

        for char1, char2 in combinations(chars, 2):

            motions1 = description[char1]
            motions2 = description[char2]
            group_results = []
            for i, mot1 in enumerate(motions1):
                for j, mot2 in enumerate(motions2):
                    group_results.append(results["motion_{}_{}_body_{}_{}".format(char1, i, char2, j)])
                    group_results.append(results["motion_{}_{}_body_{}_{}".format(char2, j, char1, i)])

            group_results = np.concatenate(group_results, axis=-1)
            mse = np.mean(group_results ** 2)
            mae = np.mean(np.abs(group_results))
            mpjpe = np.mean(np.linalg.norm(group_results, axis=1))

            out_str = "{} {} MSE {} {} (3d) = {:.6f}\n".format(args.in_dir, f"{char1}_{char2}", "(fix_hip)" if args.fix_hip else "",
                                                        "(norm_height)" if args.norm_height else "", mse)
            out_str += "{} {} MAE {} {} (3d) = {:.6f}\n".format(args.in_dir, f"{char1}_{char2}", "(fix_hip)" if args.fix_hip else "",
                                                        "(norm_height)" if args.norm_height else "", mae)
            out_str += "{} {} MPJPE {} {} (3d) = {:.6f}\n".format(args.in_dir, f"{char1}_{char2}", "(fix_hip)" if args.fix_hip else "",
                                                        "(norm_height)" if args.norm_height else "", mpjpe)

            if hasattr(args, 'out_file') and args.out_file is not None:
                with open(args.out_file, "a+") as f:
                    f.write("{:.6f} {:.6f} {:.6f}\n".format(mse, mae, mpjpe))

            print(out_str)

    group_results = [diff for _, diff in results.items()]
    group_results = np.concatenate(group_results, axis=-1)

    mse = np.mean(group_results ** 2)
    mae = np.mean(np.abs(group_results))
    mpjpe = np.mean(np.linalg.norm(group_results, axis=1))

    out_str = "{} {} MSE {} {} (3d) = {:.6f}\n".format(args.in_dir, "OVERALL", "(fix_hip)" if args.fix_hip else "",
                                                   "(norm_height)" if args.norm_height else "", mse)
    out_str += "{} {} MAE {} {} (3d) = {:.6f}\n".format(args.in_dir, "OVERALL", "(fix_hip)" if args.fix_hip else "",
                                                    "(norm_height)" if args.norm_height else "", mae)
    out_str += "{} {} MPJPE {} {} (3d) = {:.6f}\n".format(args.in_dir, "OVERALL", "(fix_hip)" if args.fix_hip else "",
                                                      "(norm_height)" if args.norm_height else "", mpjpe)

    if hasattr(args, 'out_file') and args.out_file is not None:
        with open(args.out_file, "a+") as f:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(mse, mae, mpjpe))

    print(out_str)



if __name__ == "__main__":
    main()






