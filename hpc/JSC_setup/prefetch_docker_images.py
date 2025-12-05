import glob
import os
import subprocess as sp
import argparse
from dockerfile_parse import DockerfileParser
import docker
from tqdm import tqdm


def get_baseimage(dockerfile_path):
    dfp = DockerfileParser()
    with open(dockerfile_path, "r") as f:
        dfp.content = f.read()
    return dfp.baseimage


def pull_image(image_name):
    client = docker.from_env()
    client.images.pull(image_name)

def save_image_as_tar(image_name, tar_path):
    client = docker.from_env()
    image = client.images.get(image_name)
    with open(tar_path, "wb") as f:
        for chunk in image.save(named=True):
            f.write(chunk)

def load_image(tar_path):
    cmd = ["docker", "load", "-i", tar_path]
    sp.run(cmd)

def main(args):

    tars_save_dir = args.tars_save_dir
    tbench_dir = args.tbench_dir

    tasks_dir = os.path.join(tbench_dir, "tasks")

    dockerfiles = glob.glob(os.path.join(tasks_dir, "**", "Dockerfile"), recursive=True)
    base_images = set()
    for dockerfile in dockerfiles:
        base_image = get_baseimage(dockerfile)
        base_images.add(base_image)

    print(f"Found {len(base_images)} base images.")
    print("Base images to be processed:")
    for base_image in base_images:
        print(f" - {base_image}")

    for base_image in tqdm(base_images):
        try:
            pull_image(base_image)
            save_image_as_tar(base_image, os.path.join(tars_save_dir, f"{base_image.replace('/', '_')}.tar"))
            load_image(os.path.join(tars_save_dir, f"{base_image.replace('/', '_')}.tar"))
        except Exception as e:
            print(f"[error] Failed to process {base_image}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefetch Docker images")
    parser.add_argument("--tars_save_dir", type=str, required=True, help="Directory to save tar files")
    parser.add_argument("--tbench_dir", type=str, required=True, help="Directory of the terminal benchmark")
    args = parser.parse_args()
    main(args)