import sys

import numpy as np
from numpy import linalg as LA

# from pygame import display
from lib.dct_tools import dct2contrast, dctn1, idctn1
from lib.mle import weibull  # , logistic
from lib.vid_tools import apply_frame_func
from typing import Sequence


# def get_thresholds_linear(
#     z,
#     x,
#     y,
#     ecc,
#     dct_window_size=(25, 71, 71),
#     display_peak_cpd=18.15,
#     frame_rate=120,
# ):
#     # return the thresholds for the matrix of contrast
#     # coefficients computed from the DCT transformation
#     # These values are computed from the *linear* model fitted to
#     # the data obtained from the subjective experiment
#     # OBSOLETE NOW, DO NOT USE THIS
#     coeff = np.array([0.0678, 0.0756, 0.1136, 0.1225, -0.6201])
#     thresholds = (
#         coeff[0] * logp(x / (dct_window_size[1] - 1) * display_peak_cpd)
#         + coeff[1] * logp(y / (dct_window_size[2] - 1) * display_peak_cpd)
#         + coeff[2] * logp(z / (dct_window_size[0] - 1) * frame_rate / 2)
#         + coeff[3] * logp(ecc)
#         + coeff[4]
#     )
#     return thresholds


def get_thresholds(
    z: np.array,
    x: np.array,
    y: np.array,
    ecc: float,
    dct_window_size: Sequence[int] = (25, 71, 71),
    display_peak_cpd: float = 18.15,
    frame_rate: int = 120,
    absolute_freq: bool = False
):
    # return the thresholds for the matrix of contrast
    # coefficients computed from the DCT transformation
    # These values are computed from the model fitted to
    # the data obtained from the subjective experiment

    # expects np.array of the DCT indices as inputs
    # order in paper: b_6, b_7, b_8, b_1, b_2, b_3, b_4, b_51, b_52, b_53
    # params = np.array([0.186768, 0, 0, 2.65746,  0.33332,  1.17794, 0.245735, 1.00658, 0.50214])
    # params = np.array([0, 0, 0, 3.07204, 0.509576, 0.963228, 0.225137, 1.22097, 0.45549])
    # params = np.array([0.364177, 0, 0.00627632, 2.53562, 0.353303, 0.920547, 0.179543, 1.06636, 0.395966])
    # params = np.array([0.057621, 0.000000, 0.000000, 1.056770, 0.164645, 0.951715, 0.015409, -0.134512, 0.451313, 2.262628])
    params = np.array(
        [
            0.000000,
            0.000000,
            0.000000,
            1.005115,
            0.182998,
            0.951673,
            0.017254,
            -0.137487,
            0.375285,
            2.385476,
        ]
    )
    poly_coeff = np.array([-0.255516, 0.766890, 0.382953, 3.271425])
    epsilon = 1e-6

    def polyfun(t, s):
        val = (
            poly_coeff[0] * np.power(t, 3)
            + poly_coeff[1] * np.power(t, 2)
            + poly_coeff[2] * t
            + poly_coeff[3]
        )
        idx = val < 15  # for numerical stability at large numbers
        # apply softplus to maintain non-negativity
        val[idx] = np.log(1 + np.exp(val[idx]))
        val = s * val
        return val

    def quadratic(x, a, b, c):
        return a * np.power(x, 2) + b * x + c

    def fitfun(p, x_, y_, z_, ecc_):
        # return polyfun(
        #     z_ - p[0] + p[1] * (x_ + y_) + p[2] * ecc_,
        #     p[3]
        #     - p[4] * np.power(x_ + y_ + epsilon, p[5] - p[6] * ecc_)
        #     - p[7] * np.power(ecc_, p[8])
        # )
        return polyfun(
            z_ - p[0] + p[1] * (x_ + y_) + p[2] * ecc_,
            p[3]
            - p[4] * np.power(x_ + y_, p[5])
            - p[6] * np.power(ecc_ + epsilon, quadratic(x_ + y_, p[7], p[8], p[9])),
        )

    def predictions(x_, y_, z_, ecc_):
        xp = logp(x_ / (dct_window_size[1] - 1) * display_peak_cpd)
        yp = logp(y_ / (dct_window_size[2] - 1) * display_peak_cpd)
        zp = logp(z_ / (dct_window_size[0] - 1) * frame_rate / 2)
        eccp = logp(ecc_)
        result = fitfun(params, xp, yp, zp, eccp)
        return result

    if absolute_freq:
        sensitivity = invlogp(fitfun(
            params,
            logp(x),
            logp(y),
            logp(z),
            logp(ecc)
            ))
    else:
        sensitivity = invlogp(predictions(x, y, z, ecc))
    thresholds = 1 / sensitivity
    return thresholds


def pooling(x, p=1.9932353156386882, type="minkowski", dimension="all"):
    if dimension == "all":
        if type == "smoothmax":
            scaled = p * x
            ex = np.exp(scaled)
            return np.sum(ex * x) / np.sum(ex)
        elif type == "minkowski":
            return LA.norm(x.flatten(), p)
        else:
            raise ValueError
    elif dimension == "spatial":
        raise ValueError("Not implemented yet")
    elif dimension == "temporal":
        raise ValueError("Not implemented yet")
    else:
        raise ValueError


def logp(data, lambda_1=0, lambda_2=1):
    """Box-Cox Transform to handle log-linear regression
    with zeroes"""
    if lambda_1 == 0:
        return np.log(data + lambda_2)
    else:
        return (np.power(data + lambda_2, lambda_1) - 1) / lambda_1


def invlogp(data, lambda_1=0, lambda_2=1):
    """ Inverse Box-Cox Transform """
    if lambda_1 == 0:
        return np.exp(data) - lambda_2
    else:
        return np.power(data * lambda_1 + 1, 1 / lambda_1) - lambda_2


def spatiotemp_window_visibility(
    vid_idct,
    eccentricity,
    pooling_type="minkowski",
    p=1.9932353156386882,
    gamma_fn=lambda x: x,
    display_peak_cpd=18.15,
    frame_rate=120,
):
    # scales the DCT coefficients by visibility thresholds
    # and applies pooling
    vid_dct = dctn1(apply_frame_func(vid_idct, frame_fn=gamma_fn))
    contrast = dct2contrast(vid_dct)
    window_size = contrast.shape
    zv, xv, yv = np.meshgrid(
        np.arange(window_size[0]),
        np.arange(window_size[1]),
        np.arange(window_size[2]),
        sparse=False,
        indexing="ij",
    )
    thresh = get_thresholds(
        zv,
        xv,
        yv,
        eccentricity,
        frame_rate=frame_rate,
        display_peak_cpd=display_peak_cpd,
    )
    contrast_scaled = contrast / thresh
    thresh_scaled_contrast = pooling(contrast_scaled[1:, :, :], p=p, type=pooling_type)
    return thresh_scaled_contrast


def spatiotemp_window_detection_prob(
    vid_idct,
    eccentricity,
    gamma_fn=lambda x: x,
    display_peak_cpd=18.15,
    frame_rate=120,
):
    # scales the DCT coefficients by visibility thresholds
    # and applies pooling
    pooling_type = "minkowski"
    # p = 2.05979
    p = 1.9932353156386882
    score = spatiotemp_window_visibility(
        vid_idct,
        eccentricity,
        pooling_type=pooling_type,
        p=p,
        gamma_fn=gamma_fn,
        display_peak_cpd=display_peak_cpd,
        frame_rate=frame_rate,
    )
    if score <= 0:
        return 0
    else:
        # beta_0, beta_1, guessRate, lapseRate = (
        #     -0.225834,
        #     2.290611,
        #     0.5,
        #     0.0,
        # )
        # # for logistic psychometric func
        # beta_0, beta_1, guessRate, lapseRate = (
        #     -0.17220304363275724,
        #     2.6136778540966508,
        #     0.5,
        #     0.0,
        # )
        # for weibull psychometric func
        beta_0, beta_1, guessRate, lapseRate = (
            1.7934341869413835,
            1.5000363108129804,
            0.5,
            0.0,
        )
        # prob = (
        #     logistic(score_log, beta_0, beta_1, guessRate, lapseRate) - guessRate
        # ) / (guessRate - lapseRate)  # convert to probability between [0,1]
        prob = (weibull(score, beta_0, beta_1, guessRate, lapseRate) - guessRate) / (
            (1 - guessRate) * (1 - lapseRate)
        )  # convert to probability between [0,1]
        return prob


def test_spatiotemp_visibility():
    frame = np.zeros((25, 71, 71))
    frame[0, 0, 0] = 0.5
    for eccentricity in range(40):
        frame[24, 17, 17] = 0.010
        frame_idct = idctn1(frame)
        score = spatiotemp_window_detection_prob(frame_idct, eccentricity)
        print(f"visibility {score} for eccentricity {eccentricity}")


def compute_scores(directory, eccentricity):
    import os
    import random
    import shutil

    from lib.file_tools import get_file_list
    from lib.luminance_tools import apply_gamma  # , rgb2lum
    from lib.vid_tools import npy2mp4

    files = get_file_list(directory, ext="npy")
    scores = []
    progress = ["\\", "|", "/", "-"]
    p_ind = 0
    scores = {}
    minkowski_p = 1.9932353156386882
    for e in eccentricity:
        scores[str(e)] = []
    for (n, file) in enumerate(files):
        print(f"{n} / {len(files)} " + progress[p_ind], end="\r")
        p_ind += 1
        p_ind %= len(progress)
        sys.stdout.flush()
        clip = np.load(file)
        for e in eccentricity:
            clip_score = spatiotemp_window_visibility(
                clip, e, p=minkowski_p, gamma_fn=apply_gamma
            )
            scores[str(e)].append((file, clip_score))
    print()

    limits = [
        (0.24, 0.26),
        (0.49, 0.51),
        (0.99, 1.01),
        (1.99, 2.01),
        (3.99, 4.01),
    ]
    numimgs_per_group = 3
    for e in eccentricity:
        try:
            os.mkdir(os.path.join(directory, "..", str(e)))
        except FileExistsError:
            pass
        scores_ecc = scores[str(e)]
        for lb, ub in limits:
            print(f"Scores between {lb} and {ub}:")
            prefix = f"{int((ub+lb)/2*100):03d}"
            file_list = []
            for file, score in scores_ecc:
                if lb <= score < ub:
                    print(f"{file} {score}")
                    file_list.append(file)
            random.shuffle(file_list)
            for file in file_list[0:numimgs_per_group]:
                basename = os.path.basename(file)
                shutil.copyfile(
                    file,
                    os.path.join(directory, "..", str(e), prefix + "_" + basename),
                )
            print()
        npy2mp4(os.path.join(directory, "..", str(e)))


if __name__ == "__main__":
    # get_thresholds(2, 2, 0, 15)
    # compute_scores("vid_segments/all", [10, 25, 40])
    print(get_thresholds(np.array([1]), np.array([0]), np.array([0]), np.array([0])))
    print(get_thresholds(np.array([1]), np.array([35]), np.array([35]), np.array([10])))
    test_spatiotemp_visibility()
