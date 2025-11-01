import io
import time
from typing import BinaryIO

import ibm_boto3
from ibm_botocore.client import Config

from app.config import Settings


class COSClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.cos_endpoint or not settings.cos_bucket or not settings.cos_instance_crn:
            raise ValueError(
                "Missing COS configuration. Please set COS_ENDPOINT, COS_BUCKET, and COS_INSTANCE_CRN."
            )
        # Prefer HMAC if keys are present; otherwise use IAM
        if settings.cos_hmac_access_key_id and settings.cos_hmac_secret_access_key:
            self.mode = "hmac"
            self.client = ibm_boto3.client(
                "s3",
                aws_access_key_id=settings.cos_hmac_access_key_id,
                aws_secret_access_key=settings.cos_hmac_secret_access_key,
                config=Config(signature_version="s3v4"),
                endpoint_url=settings.cos_endpoint,
            )
        else:
            self.mode = "iam"
            self.client = ibm_boto3.client(
                "s3",
                ibm_api_key_id=settings.cos_api_key,
                ibm_service_instance_id=settings.cos_instance_crn,
                ibm_auth_endpoint=settings.cos_auth_endpoint,
                config=Config(signature_version="oauth"),
                endpoint_url=settings.cos_endpoint,
            )

    def upload_fileobj(self, key: str, fileobj: BinaryIO, content_type: str = "application/pdf") -> str:
        self.client.upload_fileobj(
            Fileobj=fileobj,
            Bucket=self.settings.cos_bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )
        return f"s3://{self.settings.cos_bucket}/{key}"

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        if self.mode != "hmac":
            raise RuntimeError("Presigned URLs require HMAC credentials; IAM mode does not support presign.")
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.settings.cos_bucket, "Key": key},
            ExpiresIn=expires_in,
        )


