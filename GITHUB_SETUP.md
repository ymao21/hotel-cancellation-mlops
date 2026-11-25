# GitHub setup (repo owner: Yining)

Steps to put this project on GitHub from VS Code on a Mac.

## 0. One-time prerequisites

```bash
git --version                       # if missing, macOS will offer to install it
git config --global user.name  "Yining Mao"
git config --global user.email "<the email on your GitHub account>"
```
Optional but easiest for login: install the GitHub CLI (`brew install gh`), then run `gh auth login`
(choose GitHub.com → HTTPS → log in with a web browser).

## 1. Check the project runs locally

```bash
cd ~/Desktop/mlops/hotel-cancellation-mlops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
ruff check . && pytest              # every test should pass
```

## 2. Create the GitHub repo

**Option A: from VS Code (no terminal)**
1. Open the `hotel-cancellation-mlops` folder in VS Code.
2. Open the **Source Control** panel (Ctrl+Shift+G) → **Publish to GitHub**.
3. Sign in when prompted, pick **Private** (or Public), name it `hotel-cancellation-mlops`.
4. When VS Code asks which files to include, keep everything it suggests. `.gitignore` already
   leaves out data, `.venv` and outputs.

**Option B: terminal**
1. On github.com click **New repository**. Name: `hotel-cancellation-mlops`. **Don't** add a
   README, .gitignore or license (the project already has them). Click **Create**.
2. In the VS Code terminal:
   ```bash
   git init -b main
   git add .
   git status                       # CHECK: no .csv files and no .venv/ in the list
   git commit -m "Initial project scaffold: data, cleaning, EDA, metric, Airflow DAG, CI"
   git remote add origin https://github.com/<your-username>/hotel-cancellation-mlops.git
   git push -u origin main
   ```
   With `gh auth login` done, the push just works. Without it, GitHub won't take your account
   password: sign in through the browser pop-up or use a Personal Access Token.

Then open the repo's **Actions** tab. The **CI** workflow should turn green in about a minute.

## 3. Track the data with DVC

Do this after downloading the Kaggle CSV to `data/raw/hotel_bookings.csv`:
```bash
dvc init
dvc add data/raw/hotel_bookings.csv
git add .dvc .dvcignore data/raw/hotel_bookings.csv.dvc data/raw/.gitignore
git commit -m "Track raw data with DVC"
git push
```
Git stores only the small `.dvc` pointer file. The CSV stays out of GitHub, which is right for
data size and for Kaggle's terms.

**Sharing the data with teammates (pick one):**
- **Simplest:** each teammate downloads the CSV from Kaggle to the same path. The `.dvc`
  hash lets everyone check they have the same version (`dvc status`).
- **DVC remote on DagsHub (free):** connect the GitHub repo at dagshub.com, then follow its
  "Remote → DVC" instructions (`dvc remote add origin https://dagshub.com/<user>/hotel-cancellation-mlops.dvc`
  plus the credential commands it shows), then `dvc push`. Teammates run `dvc pull`.
  DagsHub can also host the MLflow tracking server later.

After running the pipeline, version the processed data too:
```bash
python -m src.pipeline
dvc add data/processed
git add data/processed.dvc data/.gitignore
git commit -m "Data v1: cleaned + train/test split"
git push                    # and `dvc push` if you set up a remote
```

## 4. Add teammates + protect `main`

1. Repo **Settings → Collaborators → Add people**: Akshaya, Emily, Dora (they accept the email invite).
2. **Settings → Branches (or Rules → Rulesets) → add a rule for `main`**:
   - Require a pull request before merging (1 approval)
   - Require status checks to pass → select **lint-and-test**
3. Copy the repo URL into the final slide deck (required slide 7).

## 5. Team workflow (share this with the team)

```bash
git clone https://github.com/<your-username>/hotel-cancellation-mlops.git
cd hotel-cancellation-mlops
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
# get data: dvc pull   (or download from Kaggle into data/raw/)

git checkout -b feat/mlflow-automl      # one branch per feature
# ... work, then:
ruff check . && pytest
git add -A && git commit -m "Add FLAML AutoML with MLflow tracking"
git push -u origin feat/mlflow-automl   # then open a Pull Request on GitHub
```
