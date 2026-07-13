"""
Deploy the HTML report to GitHub Pages (gh-pages branch).
"""
import os
import sys
import subprocess
import shutil
import base64

REPO = "https://G-reen-vibe:REDACTED_TOKEN@github.com/G-reen-vibe/neat-modified.git"
REPORT = "/home/z/my-project/download/report.html"
WORK = "/home/z/my-project/.zscripts/gh-pages-work"


def run(cmd, cwd=None, check=True):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and check:
        print(f"FAILED (exit {result.returncode}):", result.stderr)
        sys.exit(1)
    return result


def main():
    # clean workspace
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)

    # clone the repo (gh-pages branch if it exists, else from main)
    print("Cloning repo...")
    r = run(f"git clone --depth 1 {REPO} .", cwd=WORK, check=False)
    if r.returncode != 0:
        # try with empty init
        run("git init .", cwd=WORK)
        run(f"git remote add origin {REPO}", cwd=WORK)
        run("git fetch origin gh-pages --depth=1", cwd=WORK, check=False)
        r2 = subprocess.run("git checkout gh-pages", shell=True, cwd=WORK, capture_output=True, text=True)
        if r2.returncode != 0:
            run("git checkout -b gh-pages", cwd=WORK)
    else:
        # check if gh-pages exists
        run("git fetch origin", cwd=WORK)
        r2 = subprocess.run("git checkout gh-pages", shell=True, cwd=WORK, capture_output=True, text=True)
        if r2.returncode != 0:
            run("git checkout -b gh-pages", cwd=WORK)
        # remove all tracked files
        run("git rm -rf .", cwd=WORK, check=False)

    # copy report
    print("Copying report...")
    shutil.copy(REPORT, os.path.join(WORK, "index.html"))

    # also copy the showcase directory (for direct file access)
    print("Copying assets...")
    if os.path.exists("/home/z/my-project/download/showcase"):
        shutil.copytree("/home/z/my-project/download/showcase", os.path.join(WORK, "showcase"), dirs_exist_ok=True)
    if os.path.exists("/home/z/my-project/download/ablation"):
        shutil.copytree("/home/z/my-project/download/ablation", os.path.join(WORK, "ablation"), dirs_exist_ok=True)

    # commit + push
    print("Committing...")
    run("git add -A", cwd=WORK)
    run('git commit -m "deploy: ablation report to gh-pages"', cwd=WORK)
    print("Pushing...")
    r = run("git push origin gh-pages --force", cwd=WORK, check=False)
    if r.returncode != 0:
        print(f"Push failed: {r.stderr}")
        # try setting upstream
        run("git push -u origin gh-pages --force", cwd=WORK)

    print(f"\n✓ Report deployed to https://g-reen-vibe.github.io/neat-modified/")


if __name__ == "__main__":
    main()
