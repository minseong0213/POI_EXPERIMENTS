"""Use the orchestrator's existing R2 configuration without copying credentials."""
import configparser
import os

import boto3


def client():
    config = configparser.ConfigParser(interpolation=None)
    config.read(os.environ.get('R2_CONFIG', '/home/mlops/gpu-orchestrator/secrets/rclone.conf'))
    remote = config['r2']
    return boto3.client('s3', endpoint_url=remote['endpoint'],
                        aws_access_key_id=remote['access_key_id'],
                        aws_secret_access_key=remote['secret_access_key'],
                        region_name=remote.get('region') or 'auto')


if __name__ == '__main__':
    import json
    objects = []
    for page in client().get_paginator('list_objects_v2').paginate(Bucket='ml-experiments', Prefix='datasets/poi/'):
        objects.extend({'key': obj['Key'], 'bytes': obj['Size'], 'etag': obj['ETag']} for obj in page.get('Contents', []))
    print(json.dumps(objects, ensure_ascii=False, indent=2))
