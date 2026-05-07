"""Template for adding your own model.

A trivial right-branching baseline that attaches every token to the next one
(with relation `dep`) and the last token as root. It runs end-to-end and gives
you a starting point — replace `predict` with your real implementation.

Usage in a notebook:

    from latinbench.models.template import MyModel
    from latinbench import Bench
    bench = Bench()
    bench.run(MyModel(), force=True)   # force=True while iterating

When iterating, either bump `name` to a new slug (results are cached per name)
or pass `force=True` to skip the cache.
"""
from __future__ import annotations
from pathlib import Path

import conllu

from ..core import Model


class MyModel(Model):
    name = "my_model"  # change this when you have something real

    def predict(self, test_path: Path, out_path: Path) -> None:
        sentences = conllu.parse(test_path.read_text())
        for sent in sentences:
            self._parse_one(sent)
        out_path.write_text("".join(s.serialize() for s in sentences))

    def _parse_one(self, sent: conllu.TokenList) -> None:
        """Attach every token to the next one with `dep`; last token is root.

        Replace this with your actual parsing logic. Inputs you can rely on for
        each token: form, lemma, upos, xpos, feats. Set token['head'] (int) and
        token['deprel'] (str) for every single-word token.

        Note: CoNLL-U also has multi-word tokens (IDs like (19, '-', 20))
        which represent contractions. Those don't get HEAD/DEPREL — skip them.
        """
        single = [t for t in sent if isinstance(t["id"], int)]
        n = len(single)
        for i, tok in enumerate(single):
            if i == n - 1:
                tok["head"] = 0
                tok["deprel"] = "root"
            else:
                tok["head"] = single[i + 1]["id"]
                tok["deprel"] = "dep"
