from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="hedgeagent")
    parser.add_argument("--version", action="store_true", help="Print the package version and exit.")
    args = parser.parse_args()
    if args.version:
        from hedgeagent import __version__

        print(__version__)
        return
    parser.print_help()


if __name__ == "__main__":
    main()

