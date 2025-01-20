import argparse
import pyvista as pv


def generate_video_around_mesh(mesh_path, video_path="output_video.mp4", num_frames=100):
    # Load the mesh
    mesh = pv.read(mesh_path)

    # Initialize the plotter with the desired window size
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(video_path)

    # Add the mesh to the plotter
    plotter.add_mesh(mesh, color="white")

    # Generate an orbital path and follow it with the camera
    path = plotter.generate_orbital_path(n_points=num_frames, shift=0.0, factor=2.0)
    plotter.orbit_on_path(path, write_frames=True)
    plotter.close()


def main():
    parser = argparse.ArgumentParser(description="Generate a video surrounding a mesh using PyVista.")
    parser.add_argument("mesh_path", type=str, help="Path to the mesh file.")
    parser.add_argument(
        "--video_path", type=str, default="output_video.mp4", help="Output path for the generated video."
    )
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames in the video.")
    args = parser.parse_args()
    generate_video_around_mesh(args.mesh_path, args.video_path, args.num_frames)


if __name__ == "__main__":
    main()
