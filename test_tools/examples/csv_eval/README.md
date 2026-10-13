# CSV eval examples

Copy-ready examples of driving monocle eval test cases from a CSV with the
`monocle_csv_cases` / `load_cases_from_csv` adapter. All three are pure
`monocle_test_tools` — nothing extra to install — and all read
[`cases.example.csv`](cases.example.csv), grading against a live Okahu tenant
(they skip cleanly when `OKAHU_API_KEY` is unset).

They differ only in **how the fact-set CSV and template are supplied**:

| Example | Supplies CSV + template via | When to reach for it |
| --- | --- | --- |
| [`csv_cases_example.py`](csv_cases_example.py) | literals written in the file | simplest; start here |
| [`csv_cases_env_example.py`](csv_cases_env_example.py) | environment variables (`FACT_SET`, `TEMPLATE`) | the form CI naturally uses |
| [`csv_cases_parametrized_example.py`](csv_cases_parametrized_example.py) | pytest flags (`--fact-set`, `--template`, via [`conftest.py`](conftest.py)) | discoverable at a terminal |

**The exact `pytest` command for each is at the top of that file's docstring** —
copy it from there. All three take `OKAHU_API_KEY` from the environment — set it
inline on the command, `export` it first, or drop it in monocle's own `.env.monocle`
(current dir) or `~/.monocle/.env`, which each example loads automatically via
`monocle_apptrace` (no extra install). A plain `.env` is not auto-loaded. Each
docstring shows the forms.

Supporting files: [`cases.example.csv`](cases.example.csv) (the fact set the
examples read) and [`hallucination_test.json`](hallucination_test.json) (a custom
LLM-as-a-judge template used by the literals example).

For the CSV column schema, the label/guard-rail rules, and the underlying
concepts, see **[CSV eval test cases](../../README.md#csv-eval-test-cases)** in
the test-tools README — this folder is just runnable copies of what that section
describes.
