"""Publish a new subset prefix and verify every uploaded byte by SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path

from r2_access import client
from sample_poi import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('data/poi_34k_seed42'))
    parser.add_argument('--prefix', default='datasets/poi/subsets/poi_34k_seed42_v1/')
    args = parser.parse_args()
    prefix = args.prefix.rstrip('/') + '/'
    if not prefix.startswith('datasets/poi/subsets/'):
        raise ValueError('Destination must be under datasets/poi/subsets/')
    bucket = 'ml-experiments'
    s3 = client()
    if s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get('KeyCount', 0):
        raise FileExistsError(f'Refusing to overwrite existing prefix {prefix}')
    report = json.loads((args.directory / 'report.json').read_text())
    # The source objects currently have single-part MD5 ETags. Fail closed if changed.
    for name, source in report['sources'].items():
        head = s3.head_object(Bucket=bucket, Key='datasets/poi/' + name)
        assert head['ContentLength'] == source['bytes']
        assert head['ETag'].strip('"') == source['md5']
    files = sorted(p for p in args.directory.iterdir() if p.is_file() and p.name != '_SUCCESS.json')
    verified = {}
    for path in files:
        expected = digest(path)
        if path.name in report['outputs']:
            assert expected == report['outputs'][path.name]['sha256']
        with path.open('rb') as stream:
            s3.put_object(Bucket=bucket, Key=prefix + path.name, Body=stream,
                          IfNoneMatch='*', Metadata={'sha256': expected})
        response = s3.get_object(Bucket=bucket, Key=prefix + path.name)
        h = hashlib.sha256()
        for block in response['Body'].iter_chunks(chunk_size=1024 * 1024):
            h.update(block)
        assert h.hexdigest() == expected, path.name
        verified[path.name] = {'bytes': path.stat().st_size, 'sha256': expected}
        print(f'Uploaded and verified: {path.name}', flush=True)
    marker = json.dumps({'status': 'complete', 'verification': 'full download SHA-256',
                         'bucket': bucket, 'prefix': prefix, 'files': verified}, indent=2) + '\n'
    s3.put_object(Bucket=bucket, Key=prefix + '_SUCCESS.json', Body=marker.encode(), IfNoneMatch='*')
    assert s3.get_object(Bucket=bucket, Key=prefix + '_SUCCESS.json')['Body'].read() == marker.encode()
    (args.directory / '_SUCCESS.json').write_text(marker)
    print(f'R2 VERIFIED: r2:{bucket}/{prefix}', flush=True)


if __name__ == '__main__':
    main()
