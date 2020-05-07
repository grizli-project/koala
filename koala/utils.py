import requests
import eazy
import os


def download_remote(url, target_path, symlink_path=None, stream=True):
    response = requests.get(url, stream=stream)

    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    else:
        raise FileNotFoundError(
            f"Remote responded with error: {response.status_code}")

    if symlink_path is not None:
        # eazy.symlink_eazy_inputs()
        print(f"Symlinking between {os.path.dirname(__file__)} and ")
