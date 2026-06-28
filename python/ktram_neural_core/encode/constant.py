"""ConstantEncoder — always-on spaces; the user-level bias.

`count` always-on spaces, each emitting the same fixed `channel` regardless of input. This is
the bias mechanism: an always-on synapse that trains under the classifier's normal routine like
any other input, so it becomes a real trained offset that shifts the boundary off the origin.
`count` is the bias width (more spaces = more bias reach). There is no special update rule here;
a *different* rule for a bias subset (the deferred split-write) is a separate mechanism.
"""

from .base import AATEncoder


class ConstantEncoder(AATEncoder):
    def __init__(self, channel=0, count=1, space_size=1):
        if channel >= space_size:
            raise ValueError(f"channel {channel} out of range for space_size {space_size}")
        self.channel = channel
        self.count = count
        self._space_size = space_size

    def encode(self, value):
        return (self.channel,) * self.count

    @property
    def space_sizes(self):
        return [self._space_size] * self.count
