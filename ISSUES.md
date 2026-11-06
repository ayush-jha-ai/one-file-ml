# Good first issues

Drafts ready to be opened as GitHub issues with the `good first issue` label. Each one is scoped so that a first-time contributor can finish it in a few evenings, and each says exactly what "done" means. Read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## 1. New concept: Gaussian naive Bayes

**Labels:** `good first issue`, `new concept`

Naive Bayes is the simplest probabilistic classifier and a natural partner to logistic regression (concept 02): one models $P(\mathbf{x} \mid y)$, the other $P(y \mid \mathbf{x})$ directly.

**Tasks**
- [ ] `13-naive-bayes/model.py`: fit per-class priors, means and variances. Predict by summing log-likelihoods, using log-sum-exp for the probabilities.
- [ ] Tests against `sklearn.naive_bayes.GaussianNB`: `theta_`, `var_`, `class_prior_` and `predict_proba`. Note that scikit-learn adds `var_smoothing` times the largest feature variance to every variance; match it explicitly.
- [ ] An animation of class-conditional Gaussians and the resulting decision boundary.
- [ ] README with all eight sections. One failure mode should show what happens when features are strongly correlated (the "naive" independence assumption counts the same evidence twice).

**Done when** `make all` passes and the README's examples assert their claimed numbers.

---

## 2. Add early stopping to gradient boosting

**Labels:** `good first issue`, `enhancement`

Concept 10's first failure mode shows accuracy on new data peaking and then falling as rounds are added. The natural fix, early stopping on a validation set, is described but not implemented.

**Tasks**
- [ ] Add optional `validation_fraction` and `n_iter_no_change` parameters to `GradientBoosting` in `10-gradient-boosting/model.py`. Stay under 150 lines.
- [ ] Test against `GradientBoostingClassifier(validation_fraction=..., n_iter_no_change=...)`. Its validation split is random, so pass the same held-out rows to both, or test the stopping rule on a fixed split.
- [ ] Add a short paragraph and a runnable example to the README's "where it breaks" section showing that early stopping recovers most of the lost accuracy.

**Done when** the stopping round matches scikit-learn on a fixed split and `make all` passes.

---

## 3. New "where it breaks" example for logistic regression: correlated features

**Labels:** `good first issue`, `failure modes`

When two features are nearly copies of each other, logistic regression's individual coefficients become unstable, even though its predictions stay good. Concept 03 shows this for lasso; concept 02 does not show it yet.

**Tasks**
- [ ] Construct two features with correlation above 0.99 and refit on bootstrap resamples.
- [ ] Show, with assertions, that the spread of each coefficient across resamples is large while the spread of their sum, and of the predicted probabilities, is small.
- [ ] Add it as example 5 or replace a weaker one, keeping the section at 2 to 4 examples. Write a one-line common-mistake entry if it fits.

**Done when** the example runs from inside `02-logistic-regression/`, its printed numbers match the prose, and `make test` passes.

---

## 4. Translate a concept README

**Labels:** `good first issue`, `translation`

Many learners read technical material more comfortably in their first language.

**Tasks**
- [ ] Choose one concept (01 is a good start) and a language you write fluently.
- [ ] Create `translations/<language-code>/01-linear-regression.md` (for example `translations/es/...`).
- [ ] Translate the prose. Keep all code blocks, maths and line links unchanged so they still work, adjusting relative paths (`../../01-linear-regression/model.py#L21`).
- [ ] Add a line to the translated file naming the commit of the English README it was translated from, so later changes can be tracked.

**Done when** a second fluent speaker has reviewed it and the links resolve on GitHub.

---

## 5. Check the animations for colour-vision accessibility

**Labels:** `good first issue`, `accessibility`

Every animation uses at most three categorical colours (blue, orange, aqua), chosen to stay distinguishable for common colour-vision deficiencies. That choice has not been checked frame by frame.

**Tasks**
- [ ] Write a small script in `tools/` that loads each `demo.gif`, simulates deuteranopia, protanopia and tritanopia on a few frames, and saves a contact sheet.
- [ ] Review the sheets and list any frame where two classes or two curves become hard to tell apart.
- [ ] For each problem found, propose a fix in the relevant `animate.py`: direct labels, different marker shapes, or line styles. Colour should not be the only way to tell series apart.

**Done when** the review is posted on the issue and any fixes are merged with regenerated GIFs.
