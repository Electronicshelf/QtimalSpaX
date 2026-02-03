import os
import json
import torch
import torch.nn as nn
from Toolbox.backbone_hik import Backbone
from Toolbox.model import Generator_Rx, StyleEncoder
from misc.wing import FAN
os.environ['USE_NNPACK'] = '0'

_BACKBONE_CACHE = {}


class QUICK:
    """
    Quick metric Wrapper
    This class sets up the model architecture, manages
    data loaders, and provides utility functions for training and evaluation.

    Initialize configurations for the Backbone class.
        Args:
            config (Config): Configuration object with various settings.

    Usage:
    # Load config from YAML
    config_path = "./config.yaml"
    config = load_config(config_path)

    # Initialize the QUICK wrapper
    quick = QUICK(config)

    # Get image pairs (real and distorted) # > check ./config.yaml <
    real_image_dir = config['ref_image_dir']
    distorted_image_dir = config['distort_image_dir']
    image_size = config['image_size']
    image_pairs = get_image_pairs(real_image_dir, distorted_image_dir, image_size)

    # Compute similarity for each pair
    for real_img, dist_img, f_name in image_pairs:
        similarity_score = quick.compute_similarity(real_img, dist_img, f_name)
        print(f"{f_name} HIK score: {similarity_score}")

    """
    def __init__(self, config):
        self.config = config
        self.model_dir = config['model_save_dir']
        self.resume_iters = config['resume_iters']
        self.c_dim = config["c_dim"]
        self.c2_dim = config["c2_dim"]
        self.eps = config["eps"]
        self.style_dim = config["style_dim"]
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device selection

        # Optional quick-model load (skip heavy weights by default).
        self.load_quick_models = bool(config.get("load_quick_models", False))
        self.load_quick_weights = bool(config.get("load_quick_weights", False))
        self.G = None
        self.style_enc = None
        self.fan = None
        if self.load_quick_models:
            if self.w_hpf > 0:
                self.fan = nn.DataParallel(FAN(fname_pretrained=self.wing_path).eval())
                self.fan.get_heatmap = self.fan.module.get_heatmap

            self.G = Generator_Rx(self.image_size, self.style_dim, w_hpf=self.w_hpf)
            self.style_enc = StyleEncoder(self.image_size, self.style_dim, self.num_domains)

            self.G.to(self.device)
            if self.w_hpf > 0:
                self.fan.to(self.device)

            self.style_enc.to(self.device)
            if self.load_quick_weights:
                self.restore_model()
        self.use_cache = bool(config.get("use_cache", True))
        cache_key = self._make_cache_key(config)
        if self.use_cache and cache_key in _BACKBONE_CACHE:
            self.backbone = _BACKBONE_CACHE[cache_key]
        else:
            self.backbone = Backbone(config)
            if self.use_cache:
                _BACKBONE_CACHE[cache_key] = self.backbone

    @staticmethod
    def _make_cache_key(config):
        try:
            return json.dumps(config, sort_keys=True, default=str)
        except TypeError:
            return str(sorted(config.items()))

    def restore_model(self):
        """Restore the trained generator and discriminator."""
        if self.G is None:
            raise RuntimeError("Generator is not initialized. Set load_quick_models=True.")
        print(f'Loading the trained models from step {self.resume_iters}...')
        G_path = os.path.join(self.model_dir, f'{self.resume_iters}-G.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage), strict=False)

    def compute_hik(self, real_img_path, dist_img_path, return_maps=False):
        """
        Compute HIK similarity between image pairs.

        Returns a single similarity score by default. Set return_maps=True to
        also return the internal map vectors.
        """
        result = self.backbone.Quick_hik(real_img_path, dist_img_path)
        if isinstance(result, tuple) and len(result) == 2:
            map_ref_sq_vec, score = result
            return (map_ref_sq_vec, score) if return_maps else score
        return result
