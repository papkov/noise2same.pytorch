import numpy as np
import torch
from torch.utils.data import DataLoader

from noise2same import model
from noise2same.dataset.dummy import DummyDataset3DLarge


def test_info_padding_forward_identity(
        psf_size: int = 9, n_dim: int = 3, device: str = "cuda"
):
    # Set up identity PSF
    psf = np.zeros((psf_size,) * n_dim)
    np.put(psf, psf_size ** n_dim // 2, 1)

    # Set up identity model and optimizer
    mdl = model.Noise2Same(
        n_dim=n_dim,
        in_channels=1,
        base_channels=4,
        psf=psf,
        psf_pad_mode="constant",
        backbone=torch.nn.Identity(),
    )
    mdl.to(device)
    mdl.train()

    # Fetch dummy data batch
    dataset = DummyDataset3DLarge(n_dim=n_dim, image_size=230)
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    batch = next(iter(loader))

    padding = [
                  (b, a)
                  for b, a in zip(
            loader.dataset.tiler.margin_start,
            loader.dataset.tiler.margin_end,
        )
              ] + [(0, 0)]
    print(padding)

    x = batch["image"].to(device)
    mask = batch["mask"].to(device)

    full_size_image = np.pad(loader.dataset.image, padding)
    full_size_image = torch.from_numpy(np.moveaxis(full_size_image, -1, 0)).to(
        device
    )

    out_mask, out_raw = mdl.forward(
        x, mask, crops=batch["crop"], full_size_image=full_size_image
    )
    loss, loss_log = mdl.compute_losses_from_output(x, mask, out_mask, out_raw)

    np.isclose(loss_log["rec_mse"], 0)


def test_info_padding_forward_backpropagation(
        psf_size: int = 9, n_dim: int = 3, device: str = "cuda"
):
    torch.autograd.set_detect_anomaly(True)

    # Set up PSF
    shape = (psf_size,) * n_dim
    psf = np.random.rand(*shape)

    # Set up model and optimizer
    mdl = model.Noise2Same(
        n_dim=n_dim,
        in_channels=1,
        base_channels=4,
        psf=psf,
        psf_pad_mode="constant",
    )
    mdl.to(device)
    mdl.train()

    optimizer = torch.optim.Adam(mdl.parameters())
    optimizer.zero_grad()

    # Fetch dummy data batch
    dataset = DummyDataset3DLarge(n_dim=n_dim, image_size=256)
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    batch = next(iter(loader))

    padding = [
                  (b, a)
                  for b, a in zip(
            loader.dataset.tiler.margin_start,
            loader.dataset.tiler.margin_end,
        )
              ] + [(0, 0)]

    x = batch["image"].to(device)
    mask = batch["mask"].to(device)

    full_size_image = np.pad(loader.dataset.image, padding)
    full_size_image = torch.from_numpy(np.moveaxis(full_size_image, -1, 0)).to(
        device
    )

    out_mask, out_raw = mdl.forward(
        x, mask, crops=batch["crop"], full_size_image=full_size_image
    )
    loss, loss_log = mdl.compute_losses_from_output(x, mask, out_mask, out_raw)

    loss.backward()
    optimizer.step()
    assert torch.isnan(loss).sum().detach().cpu().numpy() == 0
