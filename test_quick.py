from Toolbox.quick_metric import QUICK
from misc.utils import load_config, get_image_pairs, save_scores_to_file


def main():
    """
    Test Script
    Perform evaluation on a dataset to measure HIK similarity
    Fetches images from the data loaders and  computes per metric
    """
    # Load config from YAML
    config_path = "./config.yaml"
    config = load_config(config_path)

    # Initialize QUICK Model
    quick = QUICK(config)

    # Get image pairs (real and distorted) # > check ./config.yaml <
    real_image_dir = config['ref_image_dir']
    distorted_image_dir = config['distort_image_dir']
    score_dir = config['test_result_dir']
    image_size = config['image_size']

    image_pairs = get_image_pairs(real_image_dir, distorted_image_dir, image_size)
    score_hik = []
    # Compute similarity for each pair
    for real_img, dist_img, f_name in image_pairs:
        similarity_score = quick.compute_hik(real_img, dist_img)
        print(f"{f_name} HIK score: {similarity_score}")
        score_hik.append(f'{f_name}: {similarity_score}')

    save_scores_to_file(score_hik, score_dir, mode="hik")


if __name__ == "__main__":
    main()
