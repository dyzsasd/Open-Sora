import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, transforms

def get_random_trajectory(seq_length, image_size, digit_size, step_length=0.05):
    "Generate a trajectory"
    canvas_size = image_size - digit_size
    x, y, v_x, v_y = np.random.random(4)
    out_x, out_y = [], []
    
    for i in range(seq_length):
        # Take a step along velocity.
        y += v_y * step_length
        x += v_x * step_length

        # Bounce off edges.
        if x <= 0:
            x = 0
            v_x = -v_x
        if x >= 1.0:
            x = 1.0
            v_x = -v_x
        if y <= 0:
            y = 0
            v_y = -v_y
        if y >= 1.0:
            y = 1.0
            v_y = -v_y
        out_x.append(x * canvas_size)
        out_y.append(y * canvas_size)

    return torch.tensor(out_x, dtype=torch.uint8), torch.tensor(out_y, dtype=torch.uint8)


def generate_moving_digit(digit_image, n_frames, image_size=None):
    digit_size = digit_image.shape[1]  # assume the digit image is square
    
    if image_size is None:
        image_size = 8 * (digit_size // 8 + 2)  # the image size should be the multiples of 8

    xs, ys = get_random_trajectory(n_frames, image_size, digit_size)
    canvas = torch.zeros((n_frames, digit_image.shape[0], image_size, image_size), dtype=torch.float)
    for i,(x,y) in enumerate(zip(xs,ys)):
        canvas[i, :, y: (y + digit_size),x: (x + digit_size)] = digit_image

    return canvas


class MnistVideoDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        n_frames = 60,
        image_size = 64,
    ):
        mnist = MNIST(root='./data', download = True, transform = ToTensor())

        self.dataset = [
            (image.repeat(3, 1, 1), label) # repeat the gray channel for creating pseudo color image of format C, H, W
            for image, label in mnist
        ]
        self.n_frames = n_frames
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)  # normalize the frames per channel
        ])
        

    def getitem(self, index):
        digit_image, label = self.dataset[index]
        sequence = generate_moving_digit(digit_image, self.n_frames, image_size=self.image_size)
        sequence = self.transform(sequence)
        sequence = sequence.permute(1, 0, 2, 3)  # TCHW -> CTHW

        return {"video": sequence, "text": label}

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.dataset)