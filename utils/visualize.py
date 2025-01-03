from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision

class Visualizer:
    def __init__(self, log_dir='./runs', model=None,input_size=(1,1,28,28)):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model = model
        self.input_size = input_size

    def log_scalar(self, tag, valuem, step):
        self.writer.add_scalar(tag, valuem, step)

    def log_image(self, tag, images, step):
        img_tensor = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, img_tensor, step)

    def log_histogram(self, model, step):
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"{name}.grad", param.grad, step)

    def log_graph(self):
        if self.model is not None:
            dummy_input = torch.zeros(self.input_size)
            self.writer.add_graph(self.model, dummy_input)

    def close(self):
        self.writer.close()
