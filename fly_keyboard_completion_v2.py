
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from flybrain import FlyBrain
except Exception:
    FlyBrain = None

warnings.filterwarnings(
    "ignore",
    message=r"CUDA path could not be detected.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Built-in demo corpus
# ---------------------------------------------------------------------------

DEMO_SENTENCES = [
    "hi this is razi speaking",
    "hello this is razi speaking",
    "hi my name is razi",
    "razi is the best",
    "razi is the better engineer",
    "razi is the person speaking",
    "this is razi speaking",
    "the car has a top speed",
    "the vehicle reached top speed",
    "increase the motor speed",
    "we need more storage space",
    "there is not enough storage space",
    "free some disk space",
    "this is a special case",
    "we need a special tool",
    "that is a special feature",
    "i am studying electrical engineering",
    "electrical engineering is interesting",
    "the electrical system failed",
    "i work with robotics",
    "i like robotics",
    "robotics is my field",
    "i wrote python code",
    "this script uses python",
    "python is useful for machine learning",
    "the cat is sleeping on the sofa",
    "i sat on the sofa",
    "the sofa is comfortable",
    "this is the beginning",
    "this is the best option",
    "this is the better choice",
    "the beach is beautiful",
    "the building is beside the road",
    "be careful with the motor",
    "because this is important",
    "before we continue check the code",
    "behind the house is a garden",
    "below the table is a box",
    "between the two options choose one",
]

# Helps candidate generation even if a word has never appeared as a target in
# the tiny demo corpus. With --corpus, the corpus vocabulary is added too.
COMMON_WORDS = """
able about above accept across action actually add after again against age ago
air all almost along already also always am among amount an and animal another
answer any anyone anything appear are area around as ask at away back bad be
beautiful became because become becomes been before began beginning behind
being below best better between big bird black blue body book both boy bring
build building business but buy by call came can car care case cat change
check child choice city class clear close code come common company continue
could country course create day did different do does dog done down during
each early easy eat electric electrical electronics electricity else end enough
even every example eye face fact family far fast feature few find first fish
fly follow food for form found four free from full game get give go good great
green group grow had hand hard has have he head help her here high him his
home house how however i idea if important in include into is it its job just
keep kind know large last later learn leave left less let life light like line
little long look made make man many may me mean more most motor move much must
my name near need never new next night no not now number of off often old on
one only open option or order other our out over own part people person place
play point possible problem program python question quick real really right
road robot robotic robotics run said same say school see seem set she should
show side simple since small so sofa software solar some something space spare
speak speaking special speed sport start state still stop storage study such
system table take than that the their them then there these they thing think
this those through time to together too top tree try two under up us use used
using very want was water way we well went were what when where which while
white who why will with word work world would write year yes you your
""".split()

WORD_RE = re.compile(r"[a-z']+")

def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())

def load_sentences(corpus: Path | None) -> list[str]:
    if corpus is None:
        return DEMO_SENTENCES

    text = corpus.read_text(encoding="utf-8", errors="ignore")
    # Split on lines and sentence punctuation. Keep only useful fragments.
    raw = re.split(r"[\n.!?]+", text)
    out = []
    for s in raw:
        toks = tokenize(s)
        if len(toks) >= 2:
            out.append(" ".join(toks))
    if not out:
        raise ValueError("Corpus contains no usable sentences.")
    return out

def split_context_and_prefix(text: str) -> tuple[list[str], str]:
    text = text.lower()
    if text and text[-1].isspace():
        return tokenize(text), ""
    words = tokenize(text)
    if not words:
        return [], ""
    return words[:-1], words[-1]

def candidates_for_prefix(prefix: str, vocab: list[str], limit: int = 100) -> list[str]:
    if not prefix:
        return vocab[:limit]
    return [w for w in vocab if w.startswith(prefix) and w != prefix][:limit]


# ---------------------------------------------------------------------------
# Baseline language score
# ---------------------------------------------------------------------------

class ContextBaseline:
    def __init__(self, sentences: list[str], alpha: float = 0.25):
        self.alpha = alpha
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)

        for sentence in sentences:
            toks = tokenize(sentence)
            for i, word in enumerate(toks):
                self.unigram[word] += 1
                if i >= 1:
                    self.bigram[toks[i - 1]][word] += 1
                if i >= 2:
                    self.trigram[(toks[i - 2], toks[i - 1])][word] += 1

    def score(self, context: list[str], candidate: str) -> float:
        u = self.unigram[candidate] + self.alpha

        b = self.alpha
        if context:
            b += self.bigram[context[-1]][candidate]

        t = self.alpha
        if len(context) >= 2:
            t += self.trigram[(context[-2], context[-1])][candidate]

        return float(np.log(u) + 1.25*np.log(b) + 2.0*np.log(t))


# ---------------------------------------------------------------------------
# Fly interface
# ---------------------------------------------------------------------------

ORN_RE = re.compile(
    r"^ORN_(DA\d+[a-z]?|DL\d+[a-z]?|DM\d+[a-z]?|DP\d+[a-z]?|"
    r"VA\d+[a-z]?|VC\d+[a-z]?|VL\d+[a-z]?|VM\d+[a-z]?|"
    r"VP\d+[a-z+]*)$"
)

def default_data_dir() -> Path:
    env = os.environ.get("FLY_DATA")
    return Path(env).expanduser() if env else Path.home() / "fly-data"

def load_metadata(data_dir: Path):
    z = np.load(data_dir / "brain.npz", allow_pickle=True)
    return {
        "cell_type": np.asarray(z["cell_type"]).astype(str),
        "superclass": np.asarray(z["superclass"]).astype(str),
    }

def discover_orn_glomeruli(meta, min_neurons=2):
    groups = {}
    for i, cell_type in enumerate(meta["cell_type"]):
        m = ORN_RE.match(str(cell_type))
        if m:
            groups.setdefault(m.group(1), []).append(i)

    groups = {
        k: np.asarray(v, dtype=np.int64)
        for k, v in groups.items()
        if len(v) >= min_neurons
    }

    if len(groups) < 20:
        raise RuntimeError(f"Only {len(groups)} usable ORN glomeruli found.")

    return dict(sorted(groups.items()))


class ORNWordEncoder:
    def __init__(self, glomeruli, glomeruli_per_word, neurons_per_glom, seed):
        self.names = list(glomeruli)
        self.groups = glomeruli
        self.glomeruli_per_word = glomeruli_per_word
        self.neurons_per_glom = neurons_per_glom
        self.seed = seed

    def glomeruli(self, word):
        local_seed = self.seed + sum((i + 1)*ord(c) for i, c in enumerate(word))
        rng = np.random.default_rng(local_seed)
        idx = rng.choice(
            len(self.names),
            size=self.glomeruli_per_word,
            replace=False,
        )
        return [self.names[i] for i in idx]

    def neurons(self, word):
        out = []
        for glom in self.glomeruli(word):
            group = self.groups[glom]
            out.extend(group[:self.neurons_per_glom].tolist())
        return np.asarray(sorted(set(out)), dtype=np.int64)


class ProjectedTrace:
    def __init__(self, n_brain, indices, dt, taus, feature_dim, seed):
        self.indices = np.asarray(indices, dtype=np.int64)

        self.slot = np.full(n_brain, -1, dtype=np.int32)
        self.slot[self.indices] = np.arange(len(self.indices), dtype=np.int32)

        rng = np.random.default_rng(seed)
        self.P = rng.normal(
            0.0,
            1.0 / np.sqrt(feature_dim),
            size=(len(self.indices), feature_dim),
        ).astype(np.float32)

        self.decays = [
            np.float32(np.exp(-dt / float(t)))
            for t in taus
        ]

        self.states = [
            np.zeros(feature_dim, dtype=np.float32)
            for _ in taus
        ]

    @property
    def dim(self):
        return sum(len(x) for x in self.states)

    def reset(self):
        for x in self.states:
            x.fill(0)

    def observe(self, fired):
        fired = np.asarray(fired, dtype=np.int64)

        slots = self.slot[fired]
        slots = slots[slots >= 0]

        contribution = self.P[slots].sum(axis=0) if len(slots) else None

        for i, decay in enumerate(self.decays):
            self.states[i] *= decay
            if contribution is not None:
                self.states[i] += contribution

    def features(self):
        return np.concatenate(self.states).astype(np.float32, copy=False)


class FlyContextModel:
    def __init__(
        self,
        device,
        data_dir,
        glomeruli_per_word,
        neurons_per_glom,
        input_strength,
        inject_steps,
        settle_steps,
        taus,
        feature_dim,
        seed,
    ):
        if FlyBrain is None:
            raise RuntimeError("flybrain is not importable.")

        self.input_strength = input_strength
        self.inject_steps = inject_steps
        self.settle_steps = settle_steps
        self.seed = seed

        meta = load_metadata(data_dir)
        glomeruli = discover_orn_glomeruli(meta)

        self.encoder = ORNWordEncoder(
            glomeruli,
            glomeruli_per_word,
            neurons_per_glom,
            seed + 1,
        )

        types = meta["cell_type"]
        kc = np.flatnonzero(
            np.array([str(x).startswith("KC") for x in types], dtype=bool)
        )

        self.brain = FlyBrain(device=device, seed=seed + 2)

        self.trace = ProjectedTrace(
            self.brain.n,
            kc,
            self.brain.dt,
            taus,
            feature_dim,
            seed + 3,
        )

    def encode_context(self, context: list[str], reset_seed: int):
        self.brain.reset(reset_seed)
        self.trace.reset()

        for word in context:
            neurons = self.encoder.neurons(word)

            for _ in range(self.inject_steps):
                fired = self.brain.step(
                    inject=[(neurons, self.input_strength)]
                )
                self.trace.observe(fired)

            for _ in range(self.settle_steps):
                fired = self.brain.step()
                self.trace.observe(fired)

        x = self.trace.features().copy()
        norm = float(np.linalg.norm(x))
        if norm > 0:
            x /= norm
        return x


# ---------------------------------------------------------------------------
# Fly retrieval memory
# ---------------------------------------------------------------------------

class FlyRetrievalMemory:
    def __init__(self, fly: FlyContextModel, cache_dir: Path):
        self.fly = fly
        self.cache_dir = cache_dir
        self.states = None
        self.targets = None
        self.contexts = None

    def make_examples(self, sentences, min_context=1, max_examples=200):
        examples = []

        for sentence in sentences:
            toks = tokenize(sentence)

            for i in range(min_context, len(toks)):
                context = toks[:i]
                target = toks[i]
                examples.append((context, target))

        # Prefer later contexts: they contain more information for the fly.
        examples.sort(key=lambda x: len(x[0]), reverse=True)
        return examples[:max_examples]

    def _cache_key(self, sentences, max_examples):
        payload = {
            "sentences": sentences,
            "max_examples": max_examples,
            "seed": self.fly.seed,
            "trace_dim": self.fly.trace.dim,
            "inject_steps": self.fly.inject_steps,
            "settle_steps": self.fly.settle_steps,
            "input_strength": self.fly.input_strength,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def fit(self, sentences, max_examples=200):
        examples = self.make_examples(
            sentences,
            max_examples=max_examples,
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        key = self._cache_key(sentences, max_examples)
        path = self.cache_dir / f"keyboard_memory_{key}.npz"

        if path.exists():
            print(f"Loading fly context cache: {path}")
            z = np.load(path, allow_pickle=False)
            self.states = np.asarray(z["states"], dtype=np.float32)
            self.targets = np.asarray(z["targets"]).astype(str)
            self.contexts = np.asarray(z["contexts"]).astype(str)
            return

        print(f"Building fly memory from {len(examples)} training contexts...")

        states = []
        targets = []
        contexts = []

        for i, (context, target) in enumerate(examples):
            if (i + 1) % 20 == 0 or i == 0 or i + 1 == len(examples):
                print(f"\r  fly contexts {i+1}/{len(examples)}", end="", flush=True)

            state = self.fly.encode_context(
                context,
                reset_seed=self.fly.seed + 10000 + i*31,
            )

            states.append(state)
            targets.append(target)
            contexts.append(" ".join(context))

        print()

        self.states = np.asarray(states, dtype=np.float32)
        self.targets = np.asarray(targets)
        self.contexts = np.asarray(contexts)

        np.savez_compressed(
            path,
            states=self.states,
            targets=self.targets,
            contexts=self.contexts,
        )

        print(f"Cached fly memory -> {path}")

    def scores(self, context, candidates, top_neighbors=15):
        if self.states is None:
            return {}, []

        q = self.fly.encode_context(
            context,
            reset_seed=self.fly.seed + 999999,
        )

        sims = self.states @ q

        k = min(top_neighbors, len(sims))
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]

        candidate_set = set(candidates)
        votes = defaultdict(float)
        neighbors = []

        for rank, j in enumerate(idx):
            sim = float(sims[j])
            target = str(self.targets[j])

            # Turn cosine similarity into a positive vote.
            weight = max(0.0, sim) ** 3

            if target in candidate_set:
                votes[target] += weight

            neighbors.append(
                (str(self.contexts[j]), target, sim)
            )

        # Normalize among candidate words only.
        total = sum(votes.values())
        if total > 0:
            votes = {w: v/total for w, v in votes.items()}
        else:
            votes = {}

        return votes, neighbors


# ---------------------------------------------------------------------------
# Completion engine
# ---------------------------------------------------------------------------

class CompletionEngine:
    def __init__(
        self,
        sentences,
        vocab,
        fly_memory=None,
        fly_weight=1.0,
    ):
        self.sentences = sentences
        self.vocab = vocab
        self.baseline = ContextBaseline(sentences)
        self.fly_memory = fly_memory
        self.fly_weight = fly_weight

    def rank(self, text, top_k=5):
        context, prefix = split_context_and_prefix(text)
        candidates = candidates_for_prefix(prefix, self.vocab, limit=200)

        if not candidates:
            return context, prefix, [], []

        fly_scores = {}
        neighbors = []

        if self.fly_memory is not None:
            fly_scores, neighbors = self.fly_memory.scores(
                context,
                candidates,
            )

        rows = []

        for word in candidates:
            base = self.baseline.score(context, word)
            fly_prob = fly_scores.get(word, 0.0)

            # Fly can boost a candidate, but lack of a stored matching target
            # does not catastrophically punish it.
            fly_bonus = np.log1p(20.0 * fly_prob)
            total = base + self.fly_weight * fly_bonus

            rows.append({
                "word": word,
                "score": float(total),
                "baseline": float(base),
                "fly_prob": float(fly_prob),
                "fly_bonus": float(fly_bonus),
            })

        rows.sort(key=lambda x: x["score"], reverse=True)

        return context, prefix, rows[:top_k], neighbors


TEST_QUERIES = [
    "hi this is razi sp",
    "the car has a top sp",
    "we need more storage sp",
    "this is a sp",
    "razi is the be",
    "i am studying elec",
    "i work with robo",
    "i wrote pyt",
    "the cat is sleeping on the so",
]


def show(engine, text, top_k, show_neighbors=False):
    context, prefix, rows, neighbors = engine.rank(text, top_k)

    print()
    print(f"> {text}")
    print(f"context: {' '.join(context)}")
    print(f"prefix:  {prefix}")

    if not rows:
        print("no candidates")
        return

    for i, row in enumerate(rows, 1):
        fly_txt = ""
        if engine.fly_memory is not None:
            fly_txt = (
                f" fly={row['fly_prob']:.3f}"
                f" bonus={row['fly_bonus']:+.3f}"
            )

        print(
            f"{i:>2}. {row['word']:<14} "
            f"score={row['score']:+.3f} "
            f"base={row['baseline']:+.3f}{fly_txt}"
        )

    if show_neighbors and neighbors:
        print("  nearest fly memories:")
        for ctx, target, sim in neighbors[:5]:
            print(f"    sim={sim:+.3f} -> {target:<12} | {ctx}")


def main():
    p = argparse.ArgumentParser(
        description="Fly-powered keyboard word-completion prototype v2."
    )

    p.add_argument("--text", default=None)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--show-neighbors", action="store_true")

    p.add_argument(
        "--fly",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--data-dir", type=Path, default=default_data_dir())

    p.add_argument("--glomeruli-per-word", type=int, default=2)
    p.add_argument("--neurons-per-glom", type=int, default=4)
    p.add_argument("--input-strength", type=float, default=0.8)
    p.add_argument("--inject-steps", type=int, default=2)
    p.add_argument("--settle-steps", type=int, default=1)
    p.add_argument("--taus", default="0.5,1.5,3.0")
    p.add_argument("--feature-dim", type=int, default=64)

    p.add_argument("--fly-weight", type=float, default=1.0)
    p.add_argument("--max-fly-contexts", type=int, default=160)
    p.add_argument("--cache-dir", type=Path, default=Path(".fly_cache"))
    p.add_argument("--seed", type=int, default=1234)

    args = p.parse_args()

    sentences = load_sentences(args.corpus)

    corpus_vocab = {
        word
        for sentence in sentences
        for word in tokenize(sentence)
    }

    vocab = sorted(corpus_vocab | set(COMMON_WORDS))

    print(f"sentences:       {len(sentences):,}")
    print(f"vocabulary:      {len(vocab):,}")

    fly_memory = None

    if args.fly:
        taus = [
            float(x.strip())
            for x in args.taus.split(",")
            if x.strip()
        ]

        print("Initializing ORN -> MaleCNS -> KC context model...")

        fly = FlyContextModel(
            device=args.device,
            data_dir=args.data_dir,
            glomeruli_per_word=args.glomeruli_per_word,
            neurons_per_glom=args.neurons_per_glom,
            input_strength=args.input_strength,
            inject_steps=args.inject_steps,
            settle_steps=args.settle_steps,
            taus=taus,
            feature_dim=args.feature_dim,
            seed=args.seed,
        )

        fly_memory = FlyRetrievalMemory(
            fly,
            cache_dir=args.cache_dir,
        )

        fly_memory.fit(
            sentences,
            max_examples=args.max_fly_contexts,
        )

    engine = CompletionEngine(
        sentences,
        vocab,
        fly_memory=fly_memory,
        fly_weight=args.fly_weight,
    )

    if args.text:
        show(
            engine,
            args.text,
            args.top_k,
            show_neighbors=args.show_neighbors,
        )
        return

    print("\nDemo queries:")

    for text in TEST_QUERIES:
        show(
            engine,
            text,
            args.top_k,
            show_neighbors=args.show_neighbors,
        )

    print("\nInteractive mode. Empty line exits.")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            break

        show(
            engine,
            text,
            args.top_k,
            show_neighbors=args.show_neighbors,
        )


if __name__ == "__main__":
    main()
