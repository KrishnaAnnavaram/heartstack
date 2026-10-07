# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of heartstack and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the heartstack documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **source** | One of the three public data sets: `hospital_india`, `kaggle_compilation`, `uci_cleveland`. | cohort (for the file), dataset (alone), site |
| **code book** | The mapping from the codes of one source to the shared labels. | dictionary, lookup |
| **harmonised table** | The table with the shared columns, the `target` and the `source` column. | merged data, clean data |
| **feature** | One of the 11 shared input columns. | variable, predictor, attribute |
| **target** | 1 for coronary artery disease, 0 for no disease. | label, outcome, class |
| **not-recorded code** | A code that the code book maps to "missing", for example slope code 0. | unknown code |
| **unknown code** | A code that is not in the code book. It stops the build. | invalid value |
| **zero-as-missing** | The rule that 0 for `cholesterol` or `resting_bp` means "not measured". | zero imputation |
| **duplicate** | A row with the same features and target as a row from another source. | copy, repeat |
| **pipeline** | The sklearn `Pipeline`: preprocessing and one model, fit as one unit. | workflow, chain |
| **missing indicator** | A 0/1 column that shows that a numeric value was missing. | flag column |
| **model** | One entry of the model zoo: `logreg`, `random_forest`, `hist_gbm`, `stack` or `xgboost`. | classifier, algorithm |
| **baseline** | The `logreg` model. Every other model must beat it. | reference model |
| **stack** | The stacking model: three base models and a logistic regression on their out-of-fold probabilities. | ensemble (alone), blend |
| **training set** | The rows that the split gives for model selection and fitting. | train data |
| **test set** | The held-out rows. heartstack scores them one time, after the selection. | validation set, hold-out (alone) |
| **outer fold** | One fold of the nested CV that scores a tuned model. | split |
| **inner search** | The `GridSearchCV` inside one outer fold that tunes the parameters. | inner loop, tuning run |
| **threshold** | The risk value above which a row is positive. It comes from out-of-fold training predictions. | cut-off, cut point |
| **operating point** | Sensitivity, specificity, PPV and NPV at the threshold. | confusion matrix (alone) |
| **calibration** | How well the predicted risk matches the observed rate: ECE, slope and intercept. | reliability |
| **decision curve** | The net benefit of the model, of "treat all" and of "treat none" at each threshold. | utility curve |
| **leave-one-source-out (LOSO)** | Train on two sources, test on the third. | cross-site validation |
| **subgroup** | A part of the test set by sex, age or source. | slice, cohort |
| **model card** | The file `model_card.md` with the use, the data, the results and the limits. | report card, fact sheet |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **harmonise** | Rename the columns and map the codes of one source to the shared labels. |
| **validate** | Apply the code book, zero-as-missing and range rules. |
| **de-duplicate** | Remove duplicates, and keep the row from the source with the higher priority. |
| **select** | Choose the model with the best mean outer-fold AUROC. |
| **tune** | Search the parameter grid with the inner search. |
| **score** | Calculate the metrics for one set of predictions. |
