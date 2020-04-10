import os

curr_dir = "/".join(os.getcwd().split("/")[:-1])
locations = {
    "datasets": {
        "intel": {
            "train": os.path.join(curr_dir, "data", "image", "train"),
            "test": os.path.join(curr_dir, "data", "image", "test"),
        }
    },
    "vggnet": os.path.join(curr_dir, "data", "image", "weights", "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"),
}
