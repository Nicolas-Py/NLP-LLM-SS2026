# What is this task, actually?

A plain-English walkthrough of what the EvaLatin 2024 Dependency Parsing models do, why Latin is interesting, and what's being classified — for someone new to dependency parsing or to Latin grammar.

For the formal task definition see the official guidelines at [`../data_and_doc/EvaLatin_2024_V11_parsing.pdf`](../data_and_doc/EvaLatin_2024_V11_parsing.pdf).

---

## What the model actually does

Dependency parsing = for every word in a sentence, the model picks **(a)** which other word is its "parent" and **(b)** what kind of relationship it has with that parent. Every sentence becomes a tree.

### A toy English example first

> *Marcus loves the girl*

| word | parent | relation |
|------|--------|----------|
| Marcus | loves | **nsubj** (subject of) |
| loves | — | **root** (the head of the sentence) |
| the | girl | **det** (determiner of) |
| girl | loves | **obj** (object of) |

That's the entire output. Two things per word: an arrow pointing up to its parent, and a label on that arrow.

```
        loves        ← root
       /     \
   Marcus   girl    ← nsubj, obj
              \
              the   ← det
```

### The same idea, in Latin

Here's a real sentence from the Tacitus test data: ***possessione et usu haud perinde afficiuntur*** ("they are not affected in the same way by ownership and use").

| ID | word | translation | parent | relation |
|----|------|-------------|--------|----------|
| 1 | possessione | "by ownership" | 6 (afficiuntur) | **obl:arg** (oblique argument — the means/agent) |
| 2 | et | "and" | 3 (usu) | **cc** (coordinating conjunction) |
| 3 | usu | "by use" | 1 (possessione) | **conj** (conjoined with possessione) |
| 4 | haud | "not" | 6 | **advmod:neg** (negation modifier) |
| 5 | perinde | "in the same way" | 6 | **advmod** (adverbial modifier) |
| 6 | afficiuntur | "are affected" | 0 | **root** |

Tree:

```
afficiuntur (root)
├─ possessione         (obl:arg — "by what?")
│  └─ usu              (conj — coordinated with possessione)
│     └─ et            (cc — the connector)
├─ haud                (advmod:neg)
└─ perinde             (advmod)
```

So out of ~50 possible relation labels, the model has to pick the right one for every word, *and* the right parent. For a 6-word sentence, that's 6 head choices × 6 label choices = a tree out of thousands of possible trees.

## What gets classified, exactly

For each token the model outputs **two predictions**:

1. **HEAD** — an integer: the ID of the parent token (or `0` if this token is the root)
2. **DEPREL** — one of ~50 labels: `nsubj`, `obj`, `obl`, `amod`, `nmod`, `advmod`, `cc`, `conj`, `det`, `case`, `mark`, `cop`, `root`, …

The other columns (form, lemma, UPOS, features) are **already given** in the test file. That's why the baseline section uses `tagger=` (empty) and only `parser=` — we're not predicting POS or lemmas, just the syntactic structure.

## Why Latin specifically is hard

In English, word order does most of the work. *"Marcus loves the girl"* ≠ *"The girl loves Marcus"* — same words, different syntax.

In Latin, **word endings** do the work. The same sentence can be written in any order:

| sentence | meaning |
|---|---|
| Marcus puellam amat | Marcus loves the girl |
| Puellam Marcus amat | Marcus loves the girl (emphasis on "the girl") |
| Amat Marcus puellam | Marcus loves the girl |
| Puellam amat Marcus | Marcus loves the girl |

All four mean exactly the same thing because:

- **Marcus** ends in *-us* → **nominative** case → it's the subject
- **puellam** ends in *-am* → **accusative** case → it's the object
- **amat** ends in *-t* → 3rd person singular verb

So the model has to learn: "look at the morphological features (case, number, gender, tense, mood…), not the word position." That's exactly why those features are pre-filled in the test files — they're crucial signal.

Now add poetry on top, where word order is even more scrambled for meter and effect, and words that go together can be separated by half a line. That's why poetry scores are lower than prose.

## What "training a parser" means

The model sees lots of (sentence, gold tree) pairs from existing UD treebanks (Perseus, PROIEL, ITTB, LLCT, UDante) and learns:

> "Given the words, their POS, and their morphological features, score every possible (head, label) combination. Pick the highest-scoring tree that's a valid tree (one root, no cycles)."

Modern parsers like LatinPipe do this with a transformer (PhilBerta) producing word embeddings, then two small classifier heads — one scoring head choices, one scoring labels — followed by a tree-decoding algorithm.

The baseline UDPipe 2 does the same thing but with a smaller, non-transformer encoder and only one treebank. Hence the ~30-point LAS gap.

## Evaluation metrics

Two main scores from the official scorer:

- **UAS** (Unlabeled Attachment Score) — % of tokens with the correct HEAD (parent), ignoring whether the label is right
- **LAS** (Labeled Attachment Score) — % of tokens with both correct HEAD *and* correct DEPREL
- **CLAS** (Content-word LAS) — like LAS, but only over content words; ignores punctuation and function words like `aux`, `case`, `cc`, `det`, `mark`. **This is the official ranking metric.**

Higher is better, max 100. The baseline lands around 50; the best system around 80.

## In one line

**The task:** given pre-tokenized, pre-tagged Latin, predict the syntactic skeleton — who modifies whom, and how.

## Next

- See real trees from the test data: [`01_explore_data.ipynb`](01_explore_data.ipynb)
- Run the baseline parser yourself: [`02_baseline_udpipe.ipynb`](02_baseline_udpipe.ipynb)
- Compare against the best system: [`03_latinpipe_compare.ipynb`](03_latinpipe_compare.ipynb)
