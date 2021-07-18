import argparse
from mygenerator.pipeline import run_generate_phone_numbers


def run_cli():
    """
    installed as `generate-phone-numbers`, is used to
    generate a dataset of images containing random sequences looking like
    phone numbers. All sequences should be unique and the script should accept the
    following parameters:
    * min spacing: minimum spacing between consecutive digits
    * max spacing: maximum spacing between consecutive digits
    * image width: width of the generated images
    * num images: number of images to generate
    * output path: where to store the generated images
    Note that besides these interfaces, you are free to implement the package as
    you wish.
        Example :
            123  456  567
            234  123  456
    """
    p = argparse.ArgumentParser(description="Generate dataset of phone number images")
    add = p.add_argument
    add("--min_spacing", type=int, default=0, help="Min spacing between digits on the image")
    add("--max_spacing", type=int, default=14, help="Max spacing between digits on the image")
    add("--image_width", type=int, default=256, help="Width of generated images")
    add("--num_images", type=int, default=1, help="n_images to generate")
    add(
        "--output_path",
        type=str,
        default="./",
        help="Output dir: images and meta.csv ",
    )
    add(
        "--config_file",
        type=str,
        default="default",
        help="config.yaml file, by default in %USER%/.mygenerator/config.yaml or default",
    )

    a = p.parse_args()

    run_generate_phone_numbers(
        num_images=a.num_images,
        min_spacing=a.min_spacing,
        max_spacing=a.max_spacing,
        image_width=a.image_width,
        output_path=a.output_path,
        config_file=a.config_file,
    )


#############################################################################
if __name__ == "__main__":
    run_cli()
