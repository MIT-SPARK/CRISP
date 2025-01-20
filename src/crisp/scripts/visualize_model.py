from torchview import draw_graph
import torch
import argparse
from crisp.models.joint import JointShapePoseNetwork


def export_to_onnx(model, input_size, filepath="joint_shape_pose.onnx"):
    model.eval()
    dummy_input = torch.randn(input_size).float().cuda()
    mask_dummy = torch.ones(input_size[0], 1, input_size[2], input_size[3]).float().cuda()
    coords_dummy = torch.randn((input_size[0], 1000, 3)).float().cuda()

    input_data = {"img": dummy_input, "mask": mask_dummy, "coords": coords_dummy}
    model_graph = draw_graph(model, input_data=input_data, device="cuda")
    model_graph.visual_graph.format = "png"
    model_graph.visual_graph.render(filename="test")
    model_graph.visual_graph

    # torch.onnx.export(
    #    model,
    #    (dummy_input, mask_dummy, coords_dummy),
    #    filepath,
    #    export_params=False,
    #    input_names=["img", "mask", "coords"],
    #    output_names=["nocs_map", "sdf", "shape_code"],
    #    dynamic_axes={
    #        "img": {0: "batch_size"},
    #        "mask": {0: "batch_size"},
    #        "coords": {0: "batch_size"},
    #        "nocs_map": {0: "batch_size"},
    #        "sdf": {0: "batch_size"},
    #        "shape_code": {0: "batch_size"},
    #    },
    # )
    # print(f"Model exported to {filepath}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export JointShapePoseNetwork to ONNX format.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model file to be loaded")
    parser.add_argument("--output-path", type=str, default="joint_shape_pose.onnx", help="Output ONNX file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Instantiate the model
    model = JointShapePoseNetwork(
        input_dim=3,
        recons_num_layers=5,
        recons_hidden_dim=256,
        backbone_model="dinov2_vits14",
        freeze_pretrained_weights=True,
        nonlinearity="sine",
        normalization_type="none",
        nocs_network_type="dpt_gnfusion_gnnocs",
        lateral_layers_type="spaced",
        backbone_input_res=(224, 224),
    )

    # Load the model weights if available from the specified model path (you may need to adjust based on format)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path)["model"])
    model.cuda()
    model.eval()

    # Define the input size (for example, batch_size x channels x height x width)
    input_size = (1, 3, 224, 224)  # Adjust input_size as needed

    # Export to ONNX
    export_to_onnx(model, input_size, args.output_path)
