"""Re-run the commands quoted in a cookbook's chapters and refresh the quoted outputs.

usage: python scripts/docs/refresh_cookbook_outputs.py <cookbook-id> [--check]

Run it with an interpreter that has website/static/cookbook-files/requirements.txt
installed. With --check it reports differences and changes nothing, which is
how CI can confirm that every quoted output still matches its script.
Runs each ```bash block that starts with "python <script>" in a temp copy of the
cookbook's files, with the venv interpreter, and replaces the ```text block that
follows (within the next 6 lines) with the actual output. Skips blocks marked
abridged (an "(abridged)" note before the block or "..." inside it).
"""

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

PY = (
    sys.executable
)  # the interpreter running this script, with the cookbook requirements installed
REPO = pathlib.Path(__file__).resolve().parents[2]
cb = sys.argv[1]
check = "--check" in sys.argv
files = REPO / "website/static/cookbook-files" / cb
tmp = pathlib.Path(tempfile.mkdtemp())
shutil.copytree(files, tmp / cb)
work = tmp / cb
BLOCK = re.compile(
    r"```bash\n(python [^\n]*(?:\\\n[^\n]*)*)\n```\n((?:[^\n]*\n){0,6}?)```text\n(.*?)\n```",
    re.DOTALL,
)
changed = 0
for mdx in sorted((REPO / "cookbook" / cb).glob("*.mdx")):
    text = mdx.read_text()

    def repl(m, mdx=mdx):
        global changed
        cmd, between, old = m.group(1), m.group(2), m.group(3)
        if (
            "abridged" in between
            or "\n..." in old
            or old.startswith("...")
            or "```" in between
        ):
            return m.group(0)
        cmdline = cmd.replace("\\\n", " ")
        script = cmdline.split()[1]
        if not (work / script).exists():
            return m.group(0)
        cmdline = PY + cmdline[len("python") :]
        res = subprocess.run(
            cmdline, shell=True, cwd=work, capture_output=True, text=True, check=False
        )
        new = (res.stdout + res.stderr).rstrip("\n")
        new = "\n".join(
            l
            for l in new.splitlines()
            if not l.startswith(("WARNING:absl", "I1", "W1"))
        )
        if "Traceback" in new:
            return m.group(0)
        if new != old:
            changed += 1
            print(
                f"{mdx.name}: output of `{cmd.splitlines()[0][:60]}` {'differs' if check else 'updated'}"
            )
            if check:
                import difflib

                for l in difflib.unified_diff(
                    old.splitlines(), new.splitlines(), lineterm="", n=0
                ):
                    print("   ", l)
                return m.group(0)
            return f"```bash\n{cmd}\n```\n{between}```text\n{new}\n```"
        return m.group(0)

    new_text = BLOCK.sub(repl, text)
    if new_text != text and not check:
        mdx.write_text(new_text)
print(f"{changed} block(s) {'differ' if check else 'updated'}")
