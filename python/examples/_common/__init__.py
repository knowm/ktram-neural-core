"""Shared, examples-only support code for the kT-RAM neural-core lessons.

Not part of the published `ktram_neural_core` package — these helpers exist only to run the
example scripts and to generate the article figures, so the lessons never re-implement the
same plumbing. Notebooks stay self-contained (they inline their helpers for Colab); the
figure-generation scripts import from here.

    from _common import experiments as ex
    from _common.plotting import plot_synapse, rails
"""
