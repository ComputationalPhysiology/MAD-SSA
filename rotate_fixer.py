from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from structlog import get_logger

import ventric_mesh.mesh_utils as mu
import meshing_utils

logger = get_logger()


def convert_pc_to_stack(pc, num_z_sections=20):
    num_points_per_z = int((pc.shape[0]) / num_z_sections)
    pc_list = []
    # Create the list of z sections
    for i in range(num_z_sections):
        pc_list.append(pc[i * num_points_per_z : (i + 1) * num_points_per_z])

    # Add the last element with the remaining points
    pc_list.append(pc[-1])

    return pc_list

def remove_duplicates(endo_points_list):
    endo_points_list_no_duplicates = []
    for points in endo_points_list:
        unique_points = points[np.diff(points, axis=0, prepend=np.nan).any(axis=1)]
        endo_points_list_no_duplicates.append(unique_points)
        if unique_points.shape[0]<points.shape[0]:
            logger.warning("Several unique points are removed...")
    return endo_points_list_no_duplicates

def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings


def main(args=None) -> int:
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data_directory",
        default="/home/shared/pcs",
        type=Path,
        help="The directory where all the patients data stored.",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default="00_results",
        type=Path,
        help="The result folder name that would be created in the directory of the sample.",
    )
    args = parser.parse_args(args)

    data_directory = args.data_directory
    output_folder = args.output_folder
    if not output_folder.exists():
        output_folder.mkdir()
    for file in sorted(data_directory.iterdir()):
        if file.suffix == '.txt':
            fname = file.as_posix()
            points = np.loadtxt(fname, delimiter=",")
            points_list = convert_pc_to_stack(points, num_z_sections=20)
            apex = points_list.pop(-1)
            points_list = remove_duplicates(points_list)
            tck_endo = mu.get_shax_from_coords(points_list, 0.0)
            ordered_points = mu.get_sample_points_from_shax(tck_endo, 40)
            outname = output_folder / file.stem
            
            # plt.scatter(points_list[0][:, 0], points_list[0][:, 1], color="k")
            # plt.scatter(points_list[0][:5, 0], points_list[0][:5, 1], color="r")
            # plt.scatter(ordered_points[0][:5, 0], ordered_points[0][:5, 1], color="g")
            # plt.savefig("test.png")
            
            ordered_points.append(apex)
            ordered_points = np.vstack(ordered_points)
            np.savetxt(outname.as_posix() + ".txt", ordered_points, delimiter=',',  fmt='%.8f')
                        
            outdir = output_folder / "Figures"
            outdir.mkdir(exist_ok=True)
            fig = go.Figure()
            for points in ordered_points:
                mu.plot_3d_points_on_figure(points, fig=fig)
            fnmae = outdir / file.stem
            fig.write_html(fnmae.as_posix() + ".html")


if __name__ == "__main__":
    main()
