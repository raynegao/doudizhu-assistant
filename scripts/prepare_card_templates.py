from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import sleep


RANK_MAP = {
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "T": "10",
    "J": "J",
    "Q": "Q",
    "K": "K",
    "A": "A",
    "2": "2",
    "X": "SJ",
    "D": "BJ",
}

DEFAULT_REPOS = (
    "tianqiraf/DouZero_For_HappyDouDiZhu@master",
    "Vincentzyx/DouZero_For_HLDDZ_FullAuto@main",
    "cyt0125/DouZero_For_Offline_Doudizhu@master",
)

CARD_TEMPLATE_RE = re.compile(r"^(?P<context>[mozcr]?[br]?)(?P<rank>[3456789TJQKA2XD])$", re.IGNORECASE)
KNOWN_PREFIXES = ("m", "o", "c", "z", "mb", "mr", "ob", "or")


@dataclass(frozen=True)
class RepoRef:
    owner_repo: str
    ref: str

    @classmethod
    def parse(cls, value: str) -> "RepoRef":
        if "@" not in value:
            raise ValueError(f"Repository must include a ref, e.g. owner/repo@main: {value}")
        owner_repo, ref = value.rsplit("@", 1)
        return cls(owner_repo=owner_repo, ref=ref)

    @property
    def slug(self) -> str:
        return self.owner_repo.replace("/", "__")


def request(url: str, retries: int = 2) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "doudizhu-assistant-data-prep"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt >= retries:
                raise
        except urllib.error.URLError:
            if attempt >= retries:
                return None
        sleep(0.3 * (attempt + 1))
    return None


def download(url: str, path: Path, strict: bool = False) -> bool:
    data = request(url)
    if data is None:
        if strict:
            raise RuntimeError(f"Failed to download: {url}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def classify_template_name(stem: str) -> str | None:
    match = CARD_TEMPLATE_RE.match(stem)
    if not match:
        return None
    return RANK_MAP[match.group("rank").upper()]


def collect_repo_templates(repo: RepoRef, output_dir: Path, dry_run: bool = False, strict: bool = False) -> int:
    count = 0
    for prefix in KNOWN_PREFIXES:
        for token, rank in RANK_MAP.items():
            name = f"{prefix}{token}.png"
            url = f"https://raw.githubusercontent.com/{repo.owner_repo}/{repo.ref}/pics/{name}"
            destination = output_dir / rank / f"{repo.slug}__{name}"
            if dry_run:
                if request(url) is None:
                    continue
                print(f"{repo.owner_repo}@{repo.ref}: {name} -> {destination}")
                count += 1
                continue
            if download(url, destination, strict=strict):
                print(f"{repo.owner_repo}@{repo.ref}: {name} -> {destination}")
                count += 1
    return count


def collect_local_templates(source_dir: Path, output_dir: Path, dry_run: bool = False) -> int:
    count = 0
    for path in sorted(source_dir.glob("*.png")):
        rank = classify_template_name(path.stem)
        if rank is None:
            continue
        destination = output_dir / rank / f"local__{path.name}"
        print(f"{path} -> {destination}")
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download DouZero fork card templates into rank folders.")
    parser.add_argument("--output-dir", default="data/cards_cls_seed", help="Seed template output directory.")
    parser.add_argument("--repo", action="append", help="Repository ref in owner/repo@branch form. Can be repeated.")
    parser.add_argument("--local-pics-dir", action="append", help="Local DouZero pics directory. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Print matched templates without downloading.")
    parser.add_argument("--strict", action="store_true", help="Fail if an expected template cannot be downloaded.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repos = tuple(RepoRef.parse(value) for value in (args.repo or DEFAULT_REPOS))
    output_dir = Path(args.output_dir)

    total = 0
    for repo in repos:
        total += collect_repo_templates(repo, output_dir, dry_run=args.dry_run, strict=args.strict)
    for local_dir in args.local_pics_dir or ():
        total += collect_local_templates(Path(local_dir), output_dir, dry_run=args.dry_run)
    print(f"matched_templates={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
