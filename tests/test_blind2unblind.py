import numpy as np
import torch
import pytest

from noise2same.denoiser.blind2unblind import Blind2Unblind, shuffle, unshuffle, mask_like


@pytest.mark.parametrize('mask_window_size', [2, 4])
def test_shuffle_unshuffle(mask_window_size: int):
    x = torch.rand(1, 1, 64, 64)
    shuffled = shuffle(x, mask_window_size)
    assert shuffled.shape == (mask_window_size ** 2, 1, 1, 64 // mask_window_size, 64 // mask_window_size)
    unshuffled = unshuffle(shuffled, mask_window_size)
    assert torch.allclose(x, unshuffled)


def test_mask_like():
    x = torch.rand(1, 1, 4, 4)
    mask = [[[[1, 0, 1, 0],
              [0, 0, 0, 0],
              [1, 0, 1, 0],
              [0, 0, 0, 0]]]]
    gen_mask = mask_like(x, mask_window_size=2, i=0)
    assert torch.allclose(gen_mask, torch.tensor(mask, dtype=torch.float32))


def test_blind2unblind():
    x = {'image': torch.rand(1, 1, 16, 16)}
    model = Blind2Unblind(mask_window_size=2)
    out = model(x['image'])
    assert out['image'].shape == (1, 1, 16, 16)
    assert out['image/masked'].shape == (1, 1, 16, 16)
    assert out['image/combined'].shape == (1, 1, 16, 16)

    loss, loss_dict = model.compute_loss(x, out)
    assert loss_dict['loss'] > 0
    assert loss_dict['loss_rev'] > 0
    assert loss_dict['loss_reg'] > 0
