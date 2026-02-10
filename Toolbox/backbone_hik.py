import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Toolbox.model import Generator_Rx, StyleEncoder
import torch
import matplotlib.pyplot as plt
from misc.wing import FAN
from misc.device import resolve_device
try:
    torch.backends.nnpack.enabled = False
except Exception:
    pass
from random import randint
import itertools

class Backbone(object):
    """
        Backbone Quick
        This class sets up the model architecture, manages data
        loaders, and provides utility functions for training and evaluation.
                Initialize configurations for the Backbone class.

            Args:
                config (Config): Configuration object with various settings.
    """

    def __init__(self, config):
        """Initialize configurations."""
        # Data loader.
        self.mask = None
        # Image size configuration.
        self.im_sz = config["image_size"]  # image resolution
        # Model configurations.
        self.c_dim = config["c_dim"]
        self.c2_dim = config["c2_dim"]
        self.style_dim = config["style_dim"]
        self.eps = config["eps"]
        self.num_domains = config["num_domains"]
        self.image_size = config["image_size"]
        self.g_conv_dim = config["g_conv_dim"]
        self.d_conv_dim = config["d_conv_dim"]
        self.g_repeat_num = config["g_repeat_num"]
        self.d_repeat_num = config["d_repeat_num"]
        self.w_hpf = config["w_hpf"]
        self.wing_path = config["wing_path"]
        # Training configurations.
        self.dataset = config["dataset"]
        self.batch_size = config["batch_size"]
        self.resume_iters = config["resume_iters"]
        # Test configurations.
        self.test_iters = config["test_iters"]
        # Miscellaneous.
        self.device = resolve_device(config.get("device"))
        # Directories.
        self.model_dir = config["model_save_dir"]
        self.output_msr = config["ref_image_dir"]
        self.test_result_dir = config["test_result_dir"]
        # Weight loading for similarity (optional).
        self.load_backbone_weights = bool(config.get("load_backbone_weights", True))
        # Histogram/pyramid knobs (optional).
        self.hist_levels = self._sanitize_hist_levels(config.get("hist_levels", [1, 2, 4, 8]))
        self.cell_stride = max(1, int(config.get("cell_stride", 1)))
        self.feature_layers = max(1, int(config.get("feature_layers", 2)))
        # Input alignment for encoder/generator compatibility.
        self.resize_input = bool(config.get("resize_input", True))
        self.resize_mode = config.get("resize_mode", "bilinear")
        # Build the model
        self.build_model()
        if self.load_backbone_weights:
            self.restore_model(self.resume_iters)

    @staticmethod
    def _sanitize_hist_levels(levels):
        if levels is None:
            return [1]
        if isinstance(levels, int):
            levels = [levels]
        if isinstance(levels, str):
            levels = [int(x) for x in levels.split(",") if x.strip()]
        clean = [max(1, int(lvl)) for lvl in levels]
        return clean or [1]

    def _resize_to_model(self, x):
        if not self.resize_input:
            return x
        target = (self.image_size, self.image_size)
        if x.shape[-2:] == target:
            return x
        mode = self.resize_mode
        if mode in ("bilinear", "bicubic", "trilinear"):
            return F.interpolate(x, size=target, mode=mode, align_corners=False)
        return F.interpolate(x, size=target, mode=mode)

    def _prepare_inputs(self, x_ref, x_dist):
        x_ref = self._resize_to_model(x_ref)
        x_dist = self._resize_to_model(x_dist)
        return x_ref, x_dist



    def build_model(self):
        """
         Instancaites model  functions
         Models: Generator,Encoder, AdaIN,  Decoder,
        """
        if self.w_hpf > 0:
            self.fan = nn.DataParallel(FAN(fname_pretrained=self.wing_path).eval())
            self.fan.get_heatmap = self.fan.module.get_heatmap

        self.G = Generator_Rx(self.image_size, self.style_dim, w_hpf=self.w_hpf)
        self.style_enc = StyleEncoder(self.image_size, self.style_dim, self.num_domains)

        self.G.to(self.device)
        if self.w_hpf > 0:
            self.fan.to(self.device)
        self.style_enc.to(self.device)


    def restore_model(self, resume_iters):
        """Restore the trained generator."""
        print(f'Loading the trained models from step {resume_iters}...')
        G_path = os.path.join(self.model_dir, f'{resume_iters}-G.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage), strict=False)


    def NormalizeData(self, vector):
        """
        :param data: in Numpy().cpu() format
        :return: Normalized value
        """
        min_val = np.min(vector)
        max_val = np.max(vector)
        scaled_vector = (vector - min_val) / (max_val - min_val)

        return scaled_vector

    def Quick_hik(self, x_real_ref, x_real_dist):
        """ Test  within pairs form  dataset."""
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        x_real_ref, x_real_dist = self._prepare_inputs(x_real_ref, x_real_dist)
        # =================================================================================== #
        #                                 Test  Begins                                         #
        # =================================================================================== #
        self.device = resolve_device(self.device)
        with torch.no_grad():
            eps = 0.0001
            ## Ref Image ##
            x_ref = x_real_ref.to(self.device)
            ## Test Image ##
            x_dist = x_real_dist.to(self.device)
            y_trg = torch.zeros(x_ref.size(0), dtype=torch.long, device=x_ref.device)
            masks_ref = self.fan.get_heatmap(x_ref) if self.w_hpf > 0 else None
            masks_dist = self.fan.get_heatmap(x_dist) if self.w_hpf > 0 else None
            s_ref = self.style_enc(x_ref, y_trg) * eps
            s_dist = self.style_enc(x_dist, y_trg) * eps
            vec_mse = self.patch_mix(x_ref, s_ref, x_dist, s_dist, masks_ref, masks_dist)
            return vec_mse.detach().cpu()

    def write_txt(self, data,  dir_p, fName, d_fName, mode="quick_" + str(randint(1, 1000))):
        """
        :param data: Array of scores
        :param dir_p: directory to save text file
        :param mode: used to as a naming scheme for different measures
        :return:
        """
        with open(dir_p + "/" + mode + ".txt", "w+") as txt_file:
            print("__________ ___________________________________")
            print("__QUICK__Quality__Score__Summary_______________")
            for i, line in enumerate(data):
                n_F = fName[i]
                d_F = d_fName[i]
                print(str(i) + " : " + n_F + " " + d_F  + " " + str(line))
                txt_file.write(n_F + " " + d_F + " " + str(line) + '\n')
                txt_file.write(n_F + " " + d_F + " " + str(line) + '\n')
                # txt_file.write(str(line[0]) + " , " + str(line[1]) + '\n')
            print("_____________________________________________")

    def view_tensor(self, tensor):
        """
        :param tensor: tensor N x C x H W
        :view: Normalized View
        """
        # print(tensor.shape)
        # exit()
        # tensor = tensor[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        tensor = tensor.detach().cpu().numpy()
        x_real = tensor[:, :]
        plt.imshow((x_real))
        plt.show()

    def flatten(self, x_f):
        # Use itertools.chain for more efficient flattening
        return list(itertools.chain.from_iterable(x_f))
    def Extract_Level_Hist(self, feature_ref, feature_test, divider, cell_stride=1):
        """
        Cells are extracted with user defined dimensions
        :param feature: Maps per convolutional layer 128, 256, 512
        :param divider: Defines the number of quadrants: 4, 8, 16
        :param cell_stride: Sample every Nth cell (1 = full grid)
        :return: total maps for each level and Histogram of each quadrant at different levels
        """
        score_list = []
        # get co-ordinate points #
        imgchl, imgheight, imgwidth = feature_ref.shape

        ######## Cell Count  ################
        cell_stride = max(1, int(cell_stride))
        # print("SANITY CHECK")
        # print(imgchl)
        # print(expected_maps)
        ####################################
        map_ref_sq_vec = []
        map_test_sq_vec = []

        cell_h = imgheight // divider
        cell_w = imgwidth // divider
        if cell_h == 0 or cell_w == 0:
            raise ValueError("divider is too large for the feature map size")

        # Number of cells along height
        # Number of cells along width
        num_cells_h = imgheight // cell_h
        num_cells_w = imgwidth // cell_w
        row_idx = range(0, num_cells_h, cell_stride)
        col_idx = range(0, num_cells_w, cell_stride)
        total_cells = len(row_idx) * len(col_idx)
        expected_maps = total_cells * imgchl

        for i in row_idx:
            for j in col_idx:

                # Extract the cell at (i, j) with size (cell_h x cell_w)
                # print(f'cell {i,j} H:{cell_h} W:{cell_w}')
                cell_ref = feature_ref[:, i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                cell_test = feature_test[:, i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]

                map_ref_sq_vec.append(torch.sum(torch.abs(cell_ref), dim=(1, 2)))
                map_test_sq_vec.append(torch.sum(torch.abs(cell_test), dim=(1, 2)))

        total_cat_r = torch.cat(map_ref_sq_vec)
        total_cat_d = torch.cat(map_test_sq_vec)
        hik_score = self.compute_hik_score(total_cat_r, total_cat_d)

        ######## Cell Count Sanity Check ##############
        # print(len(total_cat_d) , expected_maps)
        if (len(total_cat_d) != expected_maps):
            raise TypeError("Vectors must have the same length")

        score_list.append(hik_score.item())
        return map_ref_sq_vec, score_list.pop()

    @staticmethod
    def compute_hik_score(hist_1, hist_2):
        """
        :param hist_1: Sum of pixels per
        spatial map for reference
        :param hist_2: Sum of pixels per
        spatial map for distorted image
        :return: intersect
        """
        eps = 1e-10
        minima = torch.minimum(hist_1, hist_2)
        minimax = torch.maximum(hist_1, hist_2)

        # Actual intersect compare #
        intersect_1 = torch.true_divide(torch.sum(minima)+eps, torch.sum(minimax)+eps)

        # print("<<<<<<<<<<<<<<<<<<<<<intersect_1>>>>>>>>>>>>>>>>>>>>>>>")
        # print(minimax)
        # print(minima)
        # print(intersect_1)
        # print("<<<<<<<<<<<<<<<<<<<<<intersect_1>>>>>>>>>>>>>>>>>>>>>>")
        # exit()

        return intersect_1
    def w_s_s(self, vec_values, lvls):
        """" Weighted Self Similarity : WSS """
        deviation = 1 - np.std(vec_values, axis=0)
        lvls_std = [x / (x ** 2) for x in lvls] # Sample math [1/1**2, 1/2**2, 1/4**2...N] score ratio computation
        ######## Sanity Check ########
        if len(vec_values) != len(lvls_std):
            raise TypeError("Vectors must have the same length")
        ##############################
        norm_deviation = np.divide(deviation, np.sum(lvls_std))
        vec_total = []
        for i, val in enumerate(vec_values):
            l_ratio = lvls_std[i]
            vec_total.append(np.sum(np.multiply(val, l_ratio)))
        wss_quality_score = np.sum(np.multiply(vec_total, norm_deviation))
        return wss_quality_score

    @staticmethod
    def geometric_mean(vec):
        """Geometric mean computation"""
        x = np.array(vec)
        g_mean = x.prod() ** (1.0 / len(x))
        return g_mean

    def hist_int_pyramid(self, feat_1, feat_2):
        """
        Computes HIK score between histogram
        :function: compute_HIK_score computes HIK score
        :function: Extract_Level_Hist Extracts the map and histogram for each level

        :param feat_1: Feature from Convolutional layer for Good image
        :param feat_2: Feature from Convolutional layer for Distorted Image
        :return: array of HIK scores for configured levels
        """

        hik_scores = []
        feat_1_s = feat_1.squeeze(0)
        feat_2_s = feat_2.squeeze(0)
        lvls = self.hist_levels

        for divider in lvls:
            _, score = self.Extract_Level_Hist(
                feat_1_s,
                feat_2_s,
                divider,
                cell_stride=self.cell_stride
            )
            hik_scores.append(score)

        return hik_scores, lvls

    def patch_mix(self, patch, s_ref, patch_dist, s_dist, masks_ref, masks_dist):
        """ Compute Histogram pyramid and Weighted HIK score"""

        vec_hik_patch = []

        # Generator is set to Pyramid Mode
        gen_ref, *feat_ref = self.G(patch, s_ref, masks=masks_ref)
        gen_dist, *feat_dist = self.G(patch_dist, s_dist, masks=masks_dist)

        feat_pairs = [
            (f_ref, f_dist)
            for f_ref, f_dist in zip(feat_ref, feat_dist)
            if f_ref is not None and f_dist is not None
        ]
        if not feat_pairs:
            feat_pairs = [(gen_ref, gen_dist)]
        feat_pairs = feat_pairs[:self.feature_layers]

        batch = patch.size(0)
        for b in range(batch):
            wss_scores = []
            for f_ref, f_dist in feat_pairs:
                hp, lvls = self.hist_int_pyramid(f_ref[b:b+1], f_dist[b:b+1])
                wss_scores.append(self.w_s_s(hp, lvls))
            vec_hik_patch.append(self.geometric_mean(wss_scores))

        return torch.tensor(vec_hik_patch, device=patch.device, dtype=patch.dtype)
