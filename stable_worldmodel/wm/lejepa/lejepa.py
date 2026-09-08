"""LeJEPA: a passive, action-free encoder trained on designed OU pairs.

Deliberate differences from :class:`~stable_worldmodel.wm.lewm.lewm.LeWM`, all
of them load-bearing:

**No ``action_encoder``, no ``predictor``, no ``act_emb``.**
    Encoder training is passive. The whole point of the study is that the
    encoder sees *designed* latent pairs and never an action, so that whatever
    it identifies is attributable to the data's distributional structure and
    not to a prediction task. Every existing ``wm/*`` in this repo is
    action-conditioned; this is the first that is not.

**``LeJEPA`` does not satisfy the ``Dynamics`` protocol, and must not.**
    It has ``encode`` but no ``rollout``. That is not an omission to be filled
    in later -- a passive encoder has no dynamics. The stage-D model that
    wraps a frozen ``LeJEPA`` and adds a predictor is a separate class,
    :class:`~stable_worldmodel.wm.lejepa.module.FrozenEncoderWM`, so "the
    encoder is frozen before any predictor is trained" is enforced by the type
    rather than by discipline.

**The head outputs exactly ``n``**, read off the latent registry, so ``m = n``.
    V7a (``m = n - k``) and V7b (``m = n + k``) are one config field.

The objective, per the reference ``engine.py``::

    loss = lambda * SIGReg(h) + (1 - lambda) * alignment(h)

over ``h`` of shape ``(2, B, n)`` -- the two views being the two frames of an
OU pair. ``whitening_loss`` is computed and logged every step but **never
optimised**: it is the metric ``epsilon``, and the V8 sweep only means
something if it is independent of what is being minimised.
"""

import inspect

import torch
from einops import rearrange
from torch import nn


class LeJEPA(nn.Module):
    """A ViT encoder plus a head to exactly ``n`` dims.

    Args:
        encoder: Backbone taking ``(B, C, H, W)`` and returning an object with
            ``last_hidden_state``. The LeWM baseline's
            ``stable_pretraining.backbone.utils.vit_hf`` is the default choice,
            so encoder capacity is held fixed against that baseline.
        head: Maps the CLS token to ``n`` dims -- see
            :class:`~stable_worldmodel.wm.lejepa.module.NDimHead`.
        interpolate_pos_encoding: Passed to the backbone, so a resolution other
            than the pretraining one still works.
    """

    def __init__(
        self,
        encoder,
        head,
        interpolate_pos_encoding: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.interpolate_pos_encoding = interpolate_pos_encoding

        # Two backbone conventions have to work here: the HF ViT, whose forward
        # takes `interpolate_pos_encoding` and returns an object carrying
        # `last_hidden_state`, and the paper's CNN, which is a plain module
        # returning `(B, D)`. Decided once, from the signature, rather than
        # per-forward with a try/except that would also swallow real errors.
        self._takes_pos_encoding = 'interpolate_pos_encoding' in (
            inspect.signature(encoder.forward).parameters
        )

    @property
    def output_dim(self):
        """Embedding width ``m``. Equals ``n`` unless V7 is in play."""
        return self.head.output_dim

    def encode(self, info):
        """Embed pixels. No action path.

        Args:
            info: Dict with ``pixels`` of shape ``(B, T, C, H, W)``. ``T`` is
                the view axis -- 2 for an OU pair.

        Returns:
            dict: ``info`` with ``emb`` of shape ``(B, T, n)``.
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')

        if self._takes_pos_encoding:
            output = self.encoder(
                pixels, interpolate_pos_encoding=self.interpolate_pos_encoding
            )
        else:
            output = self.encoder(pixels)

        # ViT: pool the CLS token, as the LeWM baseline does. CNN: the module
        # has already pooled and projected, so its output *is* the feature.
        features = (
            output.last_hidden_state[:, 0]
            if hasattr(output, 'last_hidden_state')
            else output
        )

        info['emb'] = rearrange(self.head(features), '(b t) d -> b t d', b=b)
        return info

    @torch.no_grad()
    def embed(self, pixels):
        """Convenience path for the metric suite: pixels in, ``(B, n)`` out.

        Args:
            pixels: ``(B, C, H, W)`` or ``(B, T, C, H, W)``.

        Returns:
            torch.Tensor: ``(B, n)`` or ``(B, T, n)``.
        """
        squeeze = pixels.ndim == 4
        if squeeze:
            pixels = pixels.unsqueeze(1)
        emb = self.encode({'pixels': pixels})['emb']
        return emb[:, 0] if squeeze else emb


__all__ = ['LeJEPA']
