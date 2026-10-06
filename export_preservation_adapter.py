"""Export an explicitly selected Full research adapter without changing tensors."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Trusted local Full research .pt')
    parser.add_argument('--base-config', required=True, help='Native transformer/config.json')
    parser.add_argument('--output', required=True, help='New portable .pt (never overwritten)')
    args = parser.parse_args()
    from boogu.preservation.checkpoints import export_full_research_adapter
    config = json.loads(Path(args.base_config).read_text(encoding='utf-8'))
    print(json.dumps(export_full_research_adapter(args.source, args.output, config), indent=2))


if __name__ == '__main__':
    main()
