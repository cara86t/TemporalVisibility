import argparse
import os
from multiprocessing import Pool
from timeit import default_timer as timer

import cv2
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib import cm
from scipy.optimize import minimize_scalar

from lib.dct_tools import dct2contrast, dctn1
from lib.display_tools import eccentricity_map, get_display_params
from lib.luminance_tools import rgb2lum  # , apply_gamma, gamma22
from predictor import pooling, spatiotemp_window_detection_prob


class frame_iterator:
    def __init__(self, frames, scaled_sz, clip_sz):
        # frames are scaled to size scaled_sz
        # then clipped to clip_sz
        self.frames = frames
        self.scaled_sz = scaled_sz
        self.clip_sz = clip_sz
        self.current_frame = 0

    def __iter__(self):
        return self

    def __getitem__(self, i):
        frame = self.frames[i]
        frame_resized = cv2.resize(
            frame,
            self.scaled_sz[::-1],  # expects (w, h)
            interpolation=cv2.INTER_LINEAR,
        )
        frame_clipped = frame_resized[0 : self.clip_sz[0], 0 : self.clip_sz[1]]
        return frame_clipped

    def __len__(self):
        return self.frames.shape[0]

    def __next__(self):
        if self.current_frame < self.frames.shape[0]:
            return self.__getitem__(self.current_frame)
        else:
            raise StopIteration


def compute_visibility(window, thresholds, p=4):
    dct = dctn1(window)
    contrast = dct2contrast(dct)
    contrast_scaled = abs(contrast) / thresholds
    # compute visibility on the DCT coefficients,
    # leaving the DC component out
    visibility = pooling(contrast_scaled[1:, :, :], p=p, type="minkowski")
    return visibility


def frame_crop(frame, CROP_SIZE):
    shape = frame.shape
    crop_px = (shape[0] - CROP_SIZE[0], shape[1] - CROP_SIZE[1])
    top_crop = np.floor(crop_px[0] / 2)
    bottom_crop = crop_px[0] - top_crop
    vert_idx = (int(top_crop), int(shape[0] - bottom_crop))
    left_crop = np.floor(crop_px[1] / 2)
    right_crop = crop_px[1] - left_crop
    horz_idx = (int(left_crop), int(shape[1] - right_crop))
    if frame.ndim == 3:
        return frame[vert_idx[0] : vert_idx[1], horz_idx[0] : horz_idx[1], :]
    elif frame.ndim == 2:
        return frame[vert_idx[0] : vert_idx[1], horz_idx[0] : horz_idx[1]]
    else:
        raise ValueError


def scale_srgb_conv(srgb, t):
    scale = np.zeros(srgb.shape)
    target = np.copy(t)
    target[target < 0] = 0
    scale[srgb <= 0.04045] = target[srgb <= 0.04045]
    a = 0.055
    gamma_scale = np.power(target, 1 / 2.4)
    scale[srgb > 0.04045] = gamma_scale[srgb > 0.04045] + (
        a * (gamma_scale[srgb > 0.04045] - 1) / srgb[srgb > 0.04045]
    )
    return scale


def process_parallel_probability(args_dict):
    """Computes the probability of detecting the temporal changes
    using the perceptually calibrated model"""
    import math
    worker_id = args_dict["worker_id"]
    file_name = args_dict["FILENAME"]
    OUTPUT_DIR = args_dict["OUTPUT_DIR"]
    SKIP_NFRAMES = args_dict["SKIP_NFRAMES"] # must be accessed by worker_id = 0 only
    MAX_NFRAMES = args_dict["MAX_NFRAMES"]
    CROP_SIZE = args_dict["CROP_SIZE"]
    WINDOW_SIZE = args_dict["WINDOW_SIZE"]
    ECCENTRICITY = args_dict["ECCENTRICITY"]
    DISPLAY = args_dict["DISPLAY"]
    BASENAME = args_dict["BASENAME"]

    print(f"WORKER_{worker_id} is up!")
    clip = cv2.VideoCapture(file_name)
    if MAX_NFRAMES is not None:
        nframes = MAX_NFRAMES
    else:
        nframes = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
    if CROP_SIZE is not None:
        frame_height, frame_width = CROP_SIZE
    else:
        frame_height = int(clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(clip.get(cv2.CAP_PROP_FRAME_WIDTH))
    if FPS_OVERRIDE is not None:
        frame_rate = FPS_OVERRIDE
    else:
        frame_rate = clip.get(cv2.CAP_PROP_FPS)
    if worker_id == 0:
        print(f"[WORKER_{worker_id}] Frame rate: {frame_rate}")
    nwindows = nframes // WINDOW_SIZE[0]
    frame_jump_unit = int(math.ceil(nwindows / NPROCESS)) * WINDOW_SIZE[0]
    last_worker_frames = nframes - frame_jump_unit * (NPROCESS - 1)
    if last_worker_frames < WINDOW_SIZE[0]:
        frame_jump_unit = int(math.floor(nwindows / NPROCESS)) * WINDOW_SIZE[0]
    print(
        f"[WORKER_{worker_id}] Jumping to frame #{frame_jump_unit * worker_id + 1}/{nframes}"
    )
    clip.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * worker_id)
    temporal_slice = []
    display = get_display_params(DISPLAY)
    if ECCENTRICITY is None:
        ecc_map = eccentricity_map(display, frame_size=(frame_height, frame_width))
    else:
        ecc_map = ECCENTRICITY
    pmap_frames = []
    n = 0  # number of frames processed by the worker
    while clip.isOpened():
        ret, frame = clip.read()
        if (not ret) or (frame_jump_unit * worker_id + n >= nframes):
            # EoF
            break
        n += 1
        if (worker_id < NPROCESS - 1) and (n > frame_jump_unit):
            # if not the last worker, do not process more than
            # frame_jump_unit frames
            break
        print(
            f"[WORKER_{worker_id}] Processing frame #{frame_jump_unit * worker_id + n}..."
        )
        if CROP_SIZE is not None:
            frame = frame_crop(frame, CROP_SIZE)
        lum = rgb2lum(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float16) / 255)
        temporal_slice.append(lum)

        if len(temporal_slice) < WINDOW_SIZE[0]:
            continue
        if worker_id == 0 and SKIP_NFRAMES > 0:
            # set the first SKIP_NFRAMES of every video equal to the following frame
            if SKIP_NFRAMES > WINDOW_SIZE[0]:
                for i in range(WINDOW_SIZE[0] - 1):
                    temporal_slice[i] = temporal_slice[WINDOW_SIZE[0] - 1]
                SKIP_NFRAMES -= WINDOW_SIZE[0]
            else:
                for i in range(SKIP_NFRAMES):
                    temporal_slice[i] = temporal_slice[SKIP_NFRAMES]
                SKIP_NFRAMES = 0
        # DCT transform the frames in the temporal window
        # slice sequences of whole frame in temporal dimension
        slice_np = np.array(temporal_slice)
        pad_width = (
            (0, 0),
            (0, (WINDOW_SIZE[1] - slice_np.shape[1] % WINDOW_SIZE[1]) % WINDOW_SIZE[1]),
            (0, (WINDOW_SIZE[2] - slice_np.shape[2] % WINDOW_SIZE[2]) % WINDOW_SIZE[2]),
        )
        slice_np = np.pad(slice_np, pad_width, mode="reflect")
        pmap_frame = np.zeros(
            (
                1,
                int(
                    np.round(slice_np.shape[1] / WINDOW_SIZE[1])
                ),  # actually rounding is not required here because the frame is already padded
                int(np.round(slice_np.shape[2] / WINDOW_SIZE[2])),
            )
        )

        # compute the probability of detection maps for each frame
        for i in range(pmap_frame.shape[1]):
            idxi = (i * WINDOW_SIZE[1], (i + 1) * WINDOW_SIZE[1])
            for j in range(pmap_frame.shape[2]):
                idxj = (j * WINDOW_SIZE[2], (j + 1) * WINDOW_SIZE[2])
                if ECCENTRICITY is None:
                    eccentricity = np.mean(
                        ecc_map[idxi[0] : idxi[1], idxj[0] : idxj[1]]
                    )
                else:
                    eccentricity = ECCENTRICITY
                window = slice_np[:, idxi[0] : idxi[1], idxj[0] : idxj[1]]
                pdet = spatiotemp_window_detection_prob(
                    window,
                    eccentricity,
                    display_peak_cpd=display["PeakCPD"],
                    frame_rate=frame_rate,
                )
                pmap_frame[0, i, j] = pdet

        pmap_frames.append(pmap_frame)

        # remove the first frame to slide the window
        temporal_slice = []
    if len(pmap_frames) == 0:
        pass
    elif len(pmap_frames) == 1:
        np.save(
            os.path.join(OUTPUT_DIR, f"output_{BASENAME}_{worker_id}"), pmap_frames[0]
        )
    else:
        np.save(
            os.path.join(OUTPUT_DIR, f"output_{BASENAME}_{worker_id}"),
            np.concatenate(pmap_frames, axis=0),
        )


def save_outputs(args_dict):
    worker_id = args_dict['worker_id']
    BASENAME = args_dict['BASENAME']
    NPROCESS = args_dict['NPROCESS']
    WINDOW_SIZE = args_dict['WINDOW_SIZE']
    FILENAME = args_dict['FILENAME']
    CROP_SIZE = args_dict['CROP_SIZE']
    FPS_OVERRIDE = args_dict['FPS_OVERRIDE']
    OUTPUT_DIR = args_dict['OUTPUT_DIR']

    visualization_types = [
        "pmap_overlay",
        "pmap",
        "pmap_heatmap",
        "pmap_multiply",
    ]
    if worker_id >= len(visualization_types):
        print(f"[WORKER_{worker_id}] Nothing to do")
        return
    else:
        print(f"[WORKER_{worker_id}] Up!")
        vis_type = visualization_types[worker_id]
        pmap_frames = None
        i = 0
        while True:
            cache_file = os.path.join(OUTPUT_DIR, f"output_{BASENAME}_{i}.npy")
            if os.path.exists(cache_file):
                partial_output = np.load(
                    os.path.join(OUTPUT_DIR, f"output_{BASENAME}_{i}.npy")
                )
            else:
                break
            if pmap_frames is None:
                pmap_frames = partial_output
            else:
                pmap_frames = np.concatenate((pmap_frames, partial_output), axis=0)
            i += 1

        # plt.hist(pmap_frames.flatten(), bins=np.linspace(0, 1, 20))
        # plt.title(f"Histogram of prob. of detection for {basename}")
        # plt.savefig(f"{basename}_histogram_{worker_id}.pdf")

        # repeat the probability map frames for interpolation later
        pmap_frames = np.repeat(pmap_frames, WINDOW_SIZE[0], axis=0)

        print(f"[WORKER_{worker_id}] Linear interpolation in temporal domain...")
        # linear interpolation in the temporal domain
        for idx_frame in range(pmap_frames.shape[0]):
            if idx_frame % WINDOW_SIZE[0] == 0:
                key_frame1 = idx_frame
                key_frame2 = idx_frame + WINDOW_SIZE[0]
                if key_frame2 > pmap_frames.shape[0] - 1:
                    break
                continue
            else:
                weight1 = (key_frame2 - idx_frame) / WINDOW_SIZE[0]
                weight2 = (idx_frame - key_frame1) / WINDOW_SIZE[0]
                pmap_frames[idx_frame] = (
                    weight1 * pmap_frames[key_frame1]
                    + weight2 * pmap_frames[key_frame2]
                )

        clip = cv2.VideoCapture(FILENAME)
        if FPS_OVERRIDE is not None:
            FPS = FPS_OVERRIDE
        else:
            FPS = int(clip.get(cv2.CAP_PROP_FPS))  # output file is saved at this FPS
        if CROP_SIZE is not None:
            clip_sz = CROP_SIZE
        else:
            clip_sz = (
                int(clip.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(clip.get(cv2.CAP_PROP_FRAME_WIDTH)),
            )
        pad_width = (
            (WINDOW_SIZE[1] - clip_sz[0] % WINDOW_SIZE[1]) % WINDOW_SIZE[1],
            (WINDOW_SIZE[2] - clip_sz[1] % WINDOW_SIZE[2]) % WINDOW_SIZE[2],
        )
        scaled_sz = (clip_sz[0] + pad_width[0], clip_sz[1] + pad_width[1])
        iterator = iter(frame_iterator(pmap_frames, scaled_sz, clip_sz))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        clip = cv2.VideoCapture(FILENAME)
        if FPS_OVERRIDE is not None:
            FPS = FPS_OVERRIDE
        else:
            FPS = clip.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(
            os.path.join(OUTPUT_DIR, f"{BASENAME}_FPS{FPS}_{vis_type}.mp4"),
            fourcc,
            FPS,
            (clip_sz[1], clip_sz[0]),
        )
        n = 0
        viridis = cm.get_cmap("viridis")
        viridis = np.array(viridis.colors)

        def apply_colormap(x, colormap):
            return np.stack(
                (
                    cv2.LUT(x, colormap[:, 0]),
                    cv2.LUT(x, colormap[:, 1]),
                    cv2.LUT(x, colormap[:, 2]),
                ),
                axis=2,
            )

        # legend = np.arange(0, 255, dtype=np.uint8)
        # legend = np.reshape(legend, (1,) + legend.shape)
        # legend = np.repeat(legend, 10, axis=1)
        # legend = np.repeat(legend, 10, axis=0)
        CVT_COLOR_WARNED = False
        while clip.isOpened():
            ret, frame = clip.read()
            if not ret:
                break
            if CROP_SIZE is not None:
                frame = frame_crop(frame, CROP_SIZE)
            if worker_id == 0 and not CVT_COLOR_WARNED:
                print(f"[WORKER_{worker_id}] Converting color...")
                print(f"[WORKER_{worker_id}] If the code seems to be stuck here without any outputs in the terminal, it may be due to an issue with cv2.cvtColor taking too long. Please see Known Issues.")
                CVT_COLOR_WARNED = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if n >= len(iterator):
                break
            print(f"[WORKER_{worker_id}] Saving frame #{n + 1} of {vis_type}...")
            frame_idx = n
            frame = frame.astype(np.float32) / 255
            if vis_type == "pmap":
                pmap = iterator[frame_idx]  # this is np.float64, ndim = 2
                frame_output = pmap
            elif vis_type == "pmap_heatmap":
                pmap = np.round(iterator[frame_idx] * 255).astype(np.uint8)
                pmap_colored = apply_colormap(pmap, viridis)
                frame_output = pmap_colored
            elif vis_type == "pmap_overlay":
                pmap = np.round(iterator[frame_idx] * 255).astype(np.uint8)
                pmap_colored = apply_colormap(pmap, viridis)
                frame_output = 0.30 * frame + 0.70 * pmap_colored
            elif vis_type == "pmap_multiply":
                pmap = iterator[frame_idx]
                pmap_step = pmap.copy()
                pmap_step[pmap_step < 0.5] = 0.0
                pmap_step[pmap_step >= 0.5] = 0.5
                pmap_step += 0.5
                frame_output = np.reshape(pmap_step, pmap_step.shape + (1,)) * frame
            else:
                raise ValueError
            frame_output = np.round(frame_output * 255).astype(np.uint8)
            out.write(
                cv2.cvtColor(frame_output, cv2.COLOR_RGB2BGR)
            )  # opencv expects BGR
            n += 1
        out.release()


def main(args_dict):
    FILENAME = args_dict["FILENAME"]
    OUTPUT_DIR = args_dict["OUTPUT_DIR"]
    BASENAME = args_dict["BASENAME"]
    FPS_OVERRIDE = args_dict["FPS_OVERRIDE"]
    NPROCESS = args_dict["NPROCESS"]
    USE_CACHED = args_dict["USE_CACHED"]
    WINDOW_SIZE = args_dict["WINDOW_SIZE"]
    CROP_SIZE = args_dict["CROP_SIZE"]

    clip = cv2.VideoCapture(FILENAME)
    if FPS_OVERRIDE is not None:
        FPS = FPS_OVERRIDE
    else:
        FPS = clip.get(cv2.CAP_PROP_FPS)
    # if CROP_SIZE is not None:
    #     frame_height, frame_width = CROP_SIZE
    # else:
    #     frame_height, frame_width = (
    #         int(clip.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    #         int(clip.get(cv2.CAP_PROP_FRAME_WIDTH)))
    pool = Pool(NPROCESS)
    start = timer()
    if not USE_CACHED:
        child_args = []
        for worker_id in range(NPROCESS):
            args_copy = args_list.copy()
            args_copy['worker_id'] = worker_id
            child_args.append(args_copy)

        pool.map(process_parallel_probability, child_args)
        print(
            f"Parallel computation of pmaps completed in {(timer() - start) / 60} mins."
        )
    else:
        print(f"Using the cache of probability maps")

    # combine outputs
    pmap_frames = None
    i = 0
    while True:
        cache_file = os.path.join(OUTPUT_DIR, f"output_{BASENAME}_{i}.npy")
        if os.path.exists(cache_file):
            partial_output = np.load(cache_file)
            print(f"Loaded {cache_file}.")
        else:
            print(f"I was not able to find more cache files.")
            break
        if pmap_frames is None:
            pmap_frames = partial_output
        else:
            pmap_frames = np.concatenate((pmap_frames, partial_output), axis=0)
        i += 1

    plt.hist(pmap_frames.flatten(), bins=np.linspace(0, 1, 20))
    plt.title(f"Histogram of prob. of detection for {BASENAME}")
    plt.yscale("log")
    plt.ylim((1, 50000))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{BASENAME}_histogram.pdf"))
    pmap_frames = np.repeat(pmap_frames, WINDOW_SIZE[0], axis=0)

    # linear interpolation in the temporal domain
    for idx_frame in range(pmap_frames.shape[0]):
        if idx_frame % WINDOW_SIZE[0] == 0:
            key_frame1 = idx_frame
            key_frame2 = idx_frame + WINDOW_SIZE[0]
            if key_frame2 > pmap_frames.shape[0] - 1:
                break
            continue
        else:
            weight1 = (key_frame2 - idx_frame) / WINDOW_SIZE[0]
            weight2 = (idx_frame - key_frame1) / WINDOW_SIZE[0]
            pmap_frames[idx_frame] = (
                weight1 * pmap_frames[key_frame1] + weight2 * pmap_frames[key_frame2]
            )

    # dump the computed probabilities to file
    print("Writing probabilities to file...")
    print(f"{pmap_frames.shape}")
    with open(os.path.join(OUTPUT_DIR, f"{BASENAME}_probs.txt"), "w") as f:
        for frame in pmap_frames:
            for row in frame:
                for p in row:
                    f.write(f"{p:.5} ")
            f.write("\n")
        f.write("\n")
        # the following code writes one probability each line (if row, column, frame structure is not needed in the output txt file)
        # for p in pmap_frames.flatten():
        #     f.write(f"{p}\n")

    clip = cv2.VideoCapture(args.filename)
    if FPS_OVERRIDE is not None:
        FPS = FPS_OVERRIDE
    else:
        FPS = int(clip.get(cv2.CAP_PROP_FPS))  # output file is saved at this FPS
    if CROP_SIZE is not None:
        clip_sz = CROP_SIZE
    else:
        clip_sz = (
            int(clip.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(clip.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )

    start = timer()
    child_args = []
    print("Saving visualizations...")
    for worker_id in range(min(4, NPROCESS-1)):
        args_copy = args_dict.copy()
        args_copy['worker_id'] = worker_id
        child_args.append(args_copy)
    pool.map(save_outputs, child_args)
    print(f"Parallel saving of results completed in {(timer() - start) / 60} mins.")
    # do not uncomment the following lines unless for debugging (it's serial execution and slow)
    # args_copy = args_dict.copy()
    # args_copy['worker_id'] = 0
    # save_outputs(args_copy)
    # args_copy = args_dict.copy()
    # args_copy['worker_id'] = 1
    # save_outputs(args_copy)
    # args_copy = args_dict.copy()
    # args_copy['worker_id'] = 2
    # save_outputs(args_copy)
    # args_copy = args_dict.copy()
    # args_copy['worker_id'] = 3
    # save_outputs(args_copy)


if __name__ == "__main__":
    ## these are the defaults, most of those settings are overwritten by input args
    args_dict = {
        'NPROCESS': 8,
        'WINDOW_SIZE': (25, 71, 71),
        'FILENAME': None,
        'DISPLAY': "lg",
        'TARGET_DET_RATE': 0.30,  # for frame rate scaling
        'ECCENTRICITY': None, # assume this constant eccentricity, computed from display parameters if set to None
        'VISUALIZATION_TYPE': ("pmap_overlay", "pmap", "pmap_heatmap", "pmap_multiply"),
        'USE_CACHED': False,
        'CROP_SIZE': None,  # (2160, 3600) # for cropping bands from videos
        'FPS_OVERRIDE': 120,  # to override the FPS of the input video
        'MAX_NFRAMES': None, # process only this many frames from the beginning of the video e.g., MAX_NFRAMES = 600
        'BASENAME': None,  # set by main() for the workers
        'OUTPUT_DIR': None,  # set by main() for the workers
        'SKIP_NFRAMES': (
            None  # skip this many frames at the beginning, must be smaller than WINDOW_SIZE[0]`
        )
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        help="The video file",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="The directory for saving the results",
    )
    parser.add_argument(
        "--use-cached",
        type=bool,
        default=False,
        help="Use the intermediate outputs cached on the disk (True/False, Default: False)",
    )
    parser.add_argument(
        "--eccentricity",
        type=int,
        default=None,
        help="The eccentricity that will be used for all pixels (skips eccentricity map computation for the specified display) (Non-negative integer, Default: None)",
    )
    parser.add_argument(
        "--skip-nframes",
        type=int,
        default=0,
        help="This many of frames will be skipped at the beginning (e.g. to discard results before TAA warmup) (Non-negative integer, Default: 0)",
    )
    parser.add_argument(
        "--nprocess",
        type=int,
        default=8,
        help="Number of parallel processes, should be less than or equal to the number of processor cores, higher values increase the memory consumption (Positive Integer, Default: 8)",
    )
    parser.add_argument(
        "--max-nframes",
        type=int,
        default=None,
        help="Maximum number of frames to process from the beginning of the input video (Integer, Default: None)",
    )
    args = parser.parse_args()

    args_dict['USE_CACHED'] = args.use_cached
    args_dict['ECCENTRICITY'] = args.eccentricity
    if args_dict['ECCENTRICITY'] is not None:
        print(f"Setting eccentricity to {args_dict['ECCENTRICITY']}")
    args_dict['SKIP_NFRAMES'] = args.skip_nframes
    args_dict['NPROCESS'] = args.nprocess
    args_dict['MAX_NFRAMES'] = args.max_nframes
    args_dict['FILENAME'] = args.filename
    args_dict['OUTPUT_DIR'] = args.output_dir
    args_dict['BASENAME'] = os.path.splitext(os.path.basename(args.filename))[0]
    main(args_dict)
