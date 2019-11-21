import os
import json
import zipfile
import argparse
import requests
import io

parser = argparse.ArgumentParser()
parser.add_argument('--dest', default="./", help="destination directory folder for download")
opt = parser.parse_args()

if __name__ == "__main__":

    files = [
        "scannet_vox",
    ]

    url_download = "http://char.vc.in.tum.de/download_scan2cad/"

    dest = os.path.abspath(opt.dest)

    parameters = {"annotations": dest}
    for f in files:
        print("\n***")
        filename_dest = dest + "/" + f
        print("downloading...", f)
        res = requests.get(url_download + f + ".zip", allow_redirects=True, stream=True)
        z = zipfile.ZipFile(io.BytesIO(res.content))
        print("unzipping...", f)
        z.extractall(dest)
        if "." not in f:
            parameters[f] = dest + "/" + f
        print("saved...", filename_dest)
        print("***")

    with open("./parameters.json", 'w') as outfile:
        json.dump(parameters, outfile, indent=True)
