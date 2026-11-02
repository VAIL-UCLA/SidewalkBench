#!/usr/bin/env python3
"""Rewrite bundled asset paths from relative to absolute.

The RTX ``UsdToMdl`` material conversion does not resolve texture paths relative
to the referencing USD layer, so MDL shader inputs such as ``inputs:texture``
must point at an absolute file path. The published asset bundle stores these as
layer-relative paths (so it is relocatable), and this utility converts them to
absolute paths for the local machine after download.

Run once from the project root after downloading the asset bundle::

    uv run python -m sidewalkbench.tools.absolutize_asset_paths

The rewrite is idempotent: paths that are already absolute are left untouched,
and files with nothing to change are not rewritten.
"""

from __future__ import annotations

import argparse
import glob
import os

from pxr import Sdf, Usd, UsdUtils  # noqa: F401  (Usd import registers the crate format)

USD_SUFFIXES = ("*.usd", "*.usda", "*.usdc")


def _local_relative_refs(layer: Sdf.Layer) -> list[str]:
    """Return relative asset paths that resolve to an existing local file."""
    found: list[str] = []

    def visit(path: Sdf.Path) -> None:
        spec = layer.GetObjectAtPath(path)
        if not isinstance(spec, Sdf.AttributeSpec):
            return
        if not isinstance(spec.default, Sdf.AssetPath):
            return
        value = spec.default.path
        if not value or value.startswith("/"):
            return
        try:
            absolute = layer.ComputeAbsolutePath(value)
        except Exception:
            return
        if absolute and os.path.exists(absolute):
            found.append(value)

    layer.Traverse(layer.pseudoRoot.path, visit)
    return found


def absolutize(assets_root: str) -> tuple[int, int]:
    """Convert local relative asset paths to absolute paths under ``assets_root``.

    Args:
        assets_root: Directory containing the asset tree (usually ``assets``).

    Returns:
        Tuple of ``(files_rewritten, references_rewritten)``.
    """
    files = sorted(
        f for suffix in USD_SUFFIXES for f in glob.glob(os.path.join(assets_root, "**", suffix), recursive=True)
    )
    files_rewritten = 0
    refs_rewritten = 0

    for path in files:
        layer = Sdf.Layer.FindOrOpen(path)
        if layer is None:
            continue
        relative_refs = _local_relative_refs(layer)
        if not relative_refs:
            continue

        def resolve(value: str, layer: Sdf.Layer = layer) -> str:
            if not value or value.startswith("/"):
                return value
            try:
                absolute = layer.ComputeAbsolutePath(value)
            except Exception:
                return value
            if absolute and os.path.exists(absolute):
                return absolute
            return value

        UsdUtils.ModifyAssetPaths(layer, resolve)
        layer.Save()
        files_rewritten += 1
        refs_rewritten += len(relative_refs)

    return files_rewritten, refs_rewritten


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "assets_root",
        nargs="?",
        default="assets",
        help="Asset directory to rewrite (default: assets).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.assets_root):
        raise SystemExit(f"Asset directory not found: {args.assets_root}")

    files, refs = absolutize(args.assets_root)
    print(f"absolutized {refs} references across {files} files under {os.path.abspath(args.assets_root)}")


if __name__ == "__main__":
    main()
