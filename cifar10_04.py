import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from tqdm import tqdm

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity_clamp_no_log import SharpenedCosineSimilarity

batch_size = 100
label_smoothing = .05
max_lr = .03
n_classes = 10
n_epochs = 200
n_runs = 1
n_input_channels = 3
n_units_1 = 80
n_units_2 = 40
n_units_3 = 40

# Allow for a version to be provided at the command line, as in
if len(sys.argv) > 1:
    version = sys.argv[1]
else:
    version = "test"

# Lay out the desitinations for all the results.
accuracy_results_path = os.path.join(f"results", f"accuracy_{version}.npy")
accuracy_history_path = os.path.join(
    "results", f"accuracy_history_{version}.npy")
loss_results_path = os.path.join("results", f"loss_{version}.npy")
os.makedirs("results", exist_ok=True)

training_set = CIFAR10(
    root=os.path.join('.', 'data', 'CIFAR10'),
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            8,  # degrees rotation
            translate=(.1, .1),
            scale=(.9,1.1),
            shear=2),
        transforms.ToTensor()
    ]))
testing_set = CIFAR10(
    root=os.path.join('.', 'data', 'CIFAR10'),
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]))

training_loader = DataLoader(
    training_set,
    batch_size=batch_size,
    shuffle=True)
testing_loader = DataLoader(
    testing_set,
    batch_size=batch_size,
    shuffle=False)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.scs1 = SharpenedCosineSimilarity(
            in_channels=n_input_channels,
            out_channels=n_units_1,
            kernel_size=5)
        self.pool1 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.scs2_depth = SharpenedCosineSimilarity(
            in_channels=n_units_1,
            out_channels=n_units_1,
            kernel_size=3,
            groups=n_units_1)
        self.scs2_point = SharpenedCosineSimilarity(
            in_channels=n_units_1,
            out_channels=n_units_2,
            kernel_size=1)
        self.pool2 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.scs3 = SharpenedCosineSimilarity(
            in_channels=n_units_2,
            out_channels=n_units_3,
            kernel_size=3)
        self.pool3 = MaxAbsPool2d(kernel_size=4, stride=4, ceil_mode=True)

        self.out = nn.Linear(in_features=n_units_3, out_features=n_classes)

    def forward(self, t):
        t = self.scs1(t)
        t = self.pool1(t)

        t = self.scs2_depth(t)
        t = self.scs2_point(t)
        t = self.pool2(t)

        t = self.scs3(t)
        t = self.pool3(t)

        t = t.reshape(-1, n_units_3)
        t = self.out(t)

        return t


# Restore any previously generated results.
try:
    accuracy_results = np.load(accuracy_results_path).tolist()
    accuracy_histories = np.load(accuracy_history_path).tolist()
    loss_results = np.load(loss_results_path).tolist()
except Exception:
    loss_results = []
    accuracy_results = []
    accuracy_histories = []


steps_per_epoch = len(training_loader)

for i_run in range(n_runs):
    network = Network()

    for p in network.parameters():
        if p.requires_grad:
            print(p.numel())

    n_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Model has {n_params:_} trainable parameters.")

    try:
        os.remove("data/params/scs1_p.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs2_depth_p.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs2_point_p.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs3_p.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs1_q.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs2_depth_q.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs2_point_q.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs3_q.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs1_weights.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs2_depth_weights.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs2_point_weights.npy")
    except Exception:
        pass
    try:
        os.remove("data/params/scs3_weights.npy")
    except Exception:
        pass

    scs1_p = network.scs1.p.detach().numpy().copy().ravel()[np.newaxis, :]
    scs2_depth_p = network.scs2_depth.p.detach().numpy().copy().ravel()[
        np.newaxis, :]
    scs2_point_p = network.scs2_point.p.detach().numpy().copy().ravel()[
        np.newaxis, :]
    scs3_p = network.scs3.p.detach().numpy().copy().ravel()[np.newaxis, :]
    scs1_q = np.exp(network.scs1.log_q.detach().numpy().copy().ravel()[np.newaxis, :])
    scs2_depth_q = np.exp(network.scs2_depth.log_q.detach().numpy().copy().ravel()[
        np.newaxis, :])
    scs2_point_q = np.exp(network.scs2_point.log_q.detach().numpy().copy().ravel()[
        np.newaxis, :])
    scs3_q = np.exp(network.scs3.log_q.detach().numpy().copy().ravel()[np.newaxis, :])
    scs1_weights = network.scs1.weight.detach().numpy().copy().ravel()[np.newaxis, :]
    scs2_depth_weights = network.scs2_depth.weight.detach().numpy().copy().ravel()[
        np.newaxis, :]
    scs2_point_weights = network.scs2_point.weight.detach().numpy().copy().ravel()[
        np.newaxis, :]
    scs3_weights = network.scs3.weight.detach().numpy().copy().ravel()[np.newaxis, :]

    optimizer = optim.Adam(network.parameters(), lr=max_lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs)

    epoch_accuracy_history = []
    for i_epoch in range(n_epochs):

        epoch_start_time = time.time()
        epoch_training_loss = 0
        epoch_testing_loss = 0
        epoch_training_num_correct = 0
        epoch_testing_num_correct = 0

        with tqdm(enumerate(training_loader)) as tqdm_training_loader:
            for batch_idx, batch in tqdm_training_loader:

                images, labels = batch
                preds = network(images)
                loss = F.cross_entropy(
                    preds, labels, label_smoothing=label_smoothing)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_training_loss += loss.item() * training_loader.batch_size
                epoch_training_num_correct += (
                    preds.argmax(dim=1).eq(labels).sum().item())

                tqdm_training_loader.set_description(
                    f'Step: {batch_idx + 1}/{steps_per_epoch}, '
                    f'Epoch: {i_epoch + 1}/{n_epochs}, '
                    f'Run: {i_run + 1}/{n_runs}'
                )

        epoch_duration = time.time() - epoch_start_time
        training_loss = epoch_training_loss / len(training_loader.dataset)
        training_accuracy = (
            epoch_training_num_correct / len(training_loader.dataset))

        # At the end of each epoch run the testing data through an
        # evaluation pass to see how the model is doing.
        # Specify no_grad() to prevent a nasty out-of-memory condition.
        with torch.no_grad():
            for batch in testing_loader:
                images, labels = batch
                preds = network(images)
                loss = F.cross_entropy(preds, labels)

                epoch_testing_loss += loss.item() * testing_loader.batch_size
                epoch_testing_num_correct += (
                    preds.argmax(dim=1).eq(labels).sum().item())

            testing_loss = epoch_testing_loss / len(testing_loader.dataset)
            testing_accuracy = (
                epoch_testing_num_correct / len(testing_loader.dataset))
            epoch_accuracy_history.append(testing_accuracy)

        print(
            f"run: {i_run}   "
            f"epoch: {i_epoch}   "
            f"duration: {epoch_duration:.04}   "
            # f"learning rate: {scheduler.get_last_lr()[0]:.04}   "
            f"training loss: {training_loss:.04}   "
            f"testing loss: {testing_loss:.04}   "
            f"training accuracy: {100 * training_accuracy:.04}%   "
            f"testing accuracy: {100 * testing_accuracy:.04}%"
        )

        scs1_p = np.concatenate((
            scs1_p,
            network.scs1.p.detach().numpy().ravel()[np.newaxis, :]),
            axis=0)
        scs2_depth_p = np.concatenate((
            scs2_depth_p,
            network.scs2_depth.p.detach().numpy().ravel()[np.newaxis, :]),
            axis=0)
        scs2_point_p = np.concatenate((
            scs2_point_p,
            network.scs2_point.p.detach().numpy().ravel()[np.newaxis, :]),
            axis=0)
        scs3_p = np.concatenate((
            scs3_p,
            network.scs3.p.detach().numpy().ravel()[np.newaxis, :]),
            axis=0)
        scs1_q = np.concatenate((
            scs1_q,
            np.exp(network.scs1.log_q.detach().numpy().ravel()[np.newaxis, :])),
            axis=0)
        scs2_depth_q = np.concatenate((
            scs2_depth_q,
            np.exp(network.scs2_depth.log_q.detach().numpy().ravel()[np.newaxis, :])),
            axis=0)
        scs2_point_q = np.concatenate((
            scs2_point_q,
            np.exp(network.scs2_point.log_q.detach().numpy().ravel()[np.newaxis, :])),
            axis=0)
        scs3_q = np.concatenate((
            scs3_q,
            np.exp(network.scs3.log_q.detach().numpy().ravel()[np.newaxis, :])),
            axis=0)
        scs1_weights = np.concatenate((
            scs1_weights,
            network.scs1.weight.detach().numpy().ravel()[np.newaxis, :]),
            axis=0)
        scs2_depth_weights = np.concatenate((
            scs2_depth_weights,
            network.scs2_depth.weight.detach().numpy().ravel()[np.newaxis, :]),
            axis=0)
        scs2_point_weights = np.concatenate((
            scs2_point_weights,
            network.scs2_point.weight.detach().numpy().ravel()[np.newaxis, :]),
            axis=0)
        scs3_weights = np.concatenate((
            scs3_weights,
            network.scs3.weight.detach().numpy().ravel()[np.newaxis, :]),
            axis=0)

        np.save("data/params/scs1_p.npy", scs1_p)
        np.save("data/params/scs2_depth_p.npy", scs2_depth_p)
        np.save("data/params/scs2_point_p.npy", scs2_point_p)
        np.save("data/params/scs3_p.npy", scs3_p)
        np.save("data/params/scs1_q.npy", scs1_q)
        np.save("data/params/scs2_depth_q.npy", scs2_depth_q)
        np.save("data/params/scs2_point_q.npy", scs2_point_q)
        np.save("data/params/scs3_q.npy", scs3_q)
        np.save("data/params/scs1_weights.npy", scs1_weights)
        np.save("data/params/scs2_depth_weights.npy", scs2_depth_weights)
        np.save("data/params/scs2_point_weights.npy", scs2_point_weights)
        np.save("data/params/scs3_weights.npy", scs3_weights)

    accuracy_histories.append(epoch_accuracy_history)
    accuracy_results.append(testing_accuracy)
    loss_results.append(testing_loss)

    np.save(accuracy_history_path, np.array(accuracy_histories))
    np.save(accuracy_results_path, np.array(accuracy_results))
    np.save(loss_results_path, np.array(loss_results))
