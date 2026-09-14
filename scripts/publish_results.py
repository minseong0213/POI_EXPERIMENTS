"""Publish an immutable result bundle to R2 and verify every object by SHA-256 metadata."""
import argparse
import json
from pathlib import Path

from r2_access import client
from bundle_results import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default='artifacts/poi-validation-study-002');
    p.add_argument('--prefix',default='results/poi/validation-study-002/');args=p.parse_args()
    directory=Path(args.directory);prefix=args.prefix.rstrip('/')+'/'
    if not prefix.startswith('results/poi/'):raise ValueError('Prefix must be under results/poi/')
    s3=client();bucket='ml-experiments'
    if s3.list_objects_v2(Bucket=bucket,Prefix=prefix,MaxKeys=1).get('KeyCount',0):raise FileExistsError(f'Refusing to overwrite {prefix}')
    uploaded={}
    for path in sorted(directory.rglob('*')):
        if not path.is_file():continue
        relative=str(path.relative_to(directory));sha=digest(path);key=prefix+relative
        with path.open('rb') as stream:s3.put_object(Bucket=bucket,Key=key,Body=stream,IfNoneMatch='*',Metadata={'sha256':sha})
        head=s3.head_object(Bucket=bucket,Key=key)
        if head['ContentLength']!=path.stat().st_size or head['Metadata'].get('sha256')!=sha:raise RuntimeError(f'R2 verification failed: {key}')
        uploaded[relative]={'bytes':path.stat().st_size,'sha256':sha}
    marker=json.dumps({'status':'complete','verification':'HEAD size and SHA-256 metadata','files':uploaded},indent=2)+'\n'
    s3.put_object(Bucket=bucket,Key=prefix+'_SUCCESS.json',Body=marker.encode(),IfNoneMatch='*')
    if s3.get_object(Bucket=bucket,Key=prefix+'_SUCCESS.json')['Body'].read()!=marker.encode():raise RuntimeError('Marker verification failed')
    print(f'R2 VERIFIED: r2:{bucket}/{prefix} ({len(uploaded)} files)')


if __name__=='__main__':main()
