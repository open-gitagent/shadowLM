"""turn a document into a model that knows it — grounded, with MoRE+ routing

Point the synthesizer at reference material and it pulls out the facts, then
writes several differently-worded questions for each one. That phrasing variety
is the point: MoRE+ trains one expert per fact and routes to it with BM25 over
the *question* side, so a fact asked about only one way is an expert nobody can
reach. Answers are judged against the source passage, so the teacher can't
quietly invent things the document never said.

Run from the repo root:
    OPENAI_API_KEY=sk-... python examples/synthesize_from_doc.py
"""
from pathlib import Path

import shadowlm as slm

DOC = Path(__file__).resolve().parent / "data" / "handbook.md"
PARAPHRASES_PER_FACT = 4


def main():
    run = slm.synthesize(
        document=DOC,
        teacher=slm.synth.frontier("gpt-4o"),
        n=80,
        method="more_plus",              # → paraphrase units, one per fact
        per_scenario=PARAPHRASES_PER_FACT,
    )
    print(run.report.summary())          # the note tells you the group size

    # The rows come out grouped: PARAPHRASES_PER_FACT consecutive rows per fact,
    # which is exactly what more_plus_group_size expects.
    for row in run.dataset.rows[:PARAPHRASES_PER_FACT]:
        print("  q:", row["messages"][0]["content"])

    model = slm.load("Qwen/Qwen2.5-1.5B-Instruct")
    result = model.finetune(run.dataset, method="more_plus",
                            more_plus_group_size=PARAPHRASES_PER_FACT)
    print("final loss:", result.loss)
    model.save("out/handbook_experts", fmt="adapter")

    # ask it something phrased nothing like the document
    print(model.generate("remind me how the expense thing works?"))


if __name__ == "__main__":
    main()
