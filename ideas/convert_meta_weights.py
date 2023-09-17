import torch

def convert_meta_weights(meta_ckpt_file: str):

    meta_ckpt = torch.load(meta_ckpt_file, map_location="cpu")

    print(list(meta_ckpt.keys()))

    # print([p.dtype for p in meta_ckpt.values()])


if __name__ == "__main__":
    # convert_meta_weights("/Users/michael/llama-2/llama-2-7b/consolidated.00.pth")
    convert_meta_weights("/Users/michael/llama-2/llama-2-7b.pth")