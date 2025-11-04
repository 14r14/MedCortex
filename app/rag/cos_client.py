from typing import BinaryIO

import ibm_boto3
from ibm_botocore.client import Config

from app.config import Settings


class COSClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        if (
            not settings.cos_endpoint
            or not settings.cos_bucket
            or not settings.cos_instance_crn
        ):
            raise ValueError(
                "Missing COS configuration. Please set COS_ENDPOINT, COS_BUCKET, and COS_INSTANCE_CRN."
            )
        
        # Normalize endpoint: strip quotes, remove trailing slash, ensure https
        endpoint = settings.cos_endpoint.strip().strip('"').strip("'").rstrip('/')
        if not endpoint.startswith(('http://', 'https://')):
            endpoint = f"https://{endpoint}"
        elif endpoint.startswith('http://'):
            # Convert http to https for security
            endpoint = endpoint.replace('http://', 'https://', 1)
        
        # Prefer HMAC if keys are present; otherwise use IAM
        if settings.cos_hmac_access_key_id and settings.cos_hmac_secret_access_key:
            self.mode = "hmac"
            try:
                self.client = ibm_boto3.client(
                    "s3",
                    aws_access_key_id=settings.cos_hmac_access_key_id,
                    aws_secret_access_key=settings.cos_hmac_secret_access_key,
                    config=Config(signature_version="s3v4"),
                    endpoint_url=endpoint,
                )
            except Exception as e:
                error_msg = str(e)
                if "Invalid endpoint" in error_msg or "endpoint" in error_msg.lower():
                    raise RuntimeError(
                        f"Invalid COS endpoint format: {endpoint}\n\n"
                        "Common endpoint formats:\n"
                        "- Regional: https://s3.{region}.cloud-object-storage.appdomain.cloud\n"
                        "- Cross-region: https://s3.cloud-object-storage.appdomain.cloud\n"
                        "- Private: https://s3.{region}.private.cloud-object-storage.appdomain.cloud\n\n"
                        "Please verify your COS_ENDPOINT environment variable.\n"
                        f"Original error: {error_msg}"
                    ) from e
                raise RuntimeError(
                    f"Failed to initialize COS client with HMAC authentication: {e}\n"
                    "Please check your network connection and COS credentials."
                ) from e
        else:
            self.mode = "iam"
            if not settings.cos_api_key:
                raise ValueError(
                    "Missing IBM Cloud API key. Please set COS_API_KEY or IBM_CLOUD_API_KEY."
                )
            try:
                self.client = ibm_boto3.client(
                    "s3",
                    ibm_api_key_id=settings.cos_api_key,
                    ibm_service_instance_id=settings.cos_instance_crn,
                    ibm_auth_endpoint=settings.cos_auth_endpoint,
                    config=Config(signature_version="oauth"),
                    endpoint_url=endpoint,
                )
            except Exception as e:
                error_msg = str(e)
                if (
                    "Failed to resolve" in error_msg
                    or "NameResolutionError" in error_msg
                    or "nodename nor servname" in error_msg
                ):
                    raise RuntimeError(
                        f"Network connectivity error: Cannot reach IBM Cloud IAM service ({settings.cos_auth_endpoint})\n\n"
                        "Troubleshooting steps:\n"
                        "1. Check your internet connection\n"
                        "2. Verify DNS resolution (try: nslookup iam.cloud.ibm.com or ping iam.cloud.ibm.com)\n"
                        "3. Check if you're behind a firewall/proxy that blocks IBM Cloud services\n"
                        "4. If using VPN, ensure it allows access to IBM Cloud endpoints\n"
                        "5. Try using HMAC authentication instead by setting COS_HMAC_ACCESS_KEY_ID and COS_HMAC_SECRET_ACCESS_KEY\n\n"
                        f"Original error: {error_msg}"
                    ) from e
                if "Invalid endpoint" in error_msg or "endpoint" in error_msg.lower():
                    raise RuntimeError(
                        f"Invalid COS endpoint format: {endpoint}\n\n"
                        "Common endpoint formats:\n"
                        "- Regional: https://s3.{region}.cloud-object-storage.appdomain.cloud\n"
                        "- Cross-region: https://s3.cloud-object-storage.appdomain.cloud\n"
                        "- Private: https://s3.{region}.private.cloud-object-storage.appdomain.cloud\n\n"
                        "Please verify your COS_ENDPOINT environment variable.\n"
                        f"Original error: {error_msg}"
                    ) from e
                raise RuntimeError(
                    f"Failed to initialize COS client with IAM authentication: {e}\n"
                    "Please check your network connection, IBM Cloud API key, and COS configuration.\n"
                    f"Endpoint used: {endpoint}\n"
                    f"Bucket: {settings.cos_bucket}\n"
                    f"Instance CRN: {settings.cos_instance_crn[:50]}..."
                ) from e

    def upload_fileobj(
        self, key: str, fileobj: BinaryIO, content_type: str = "application/pdf"
    ) -> str:
        self.client.upload_fileobj(
            Fileobj=fileobj,
            Bucket=self.settings.cos_bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )
        return f"s3://{self.settings.cos_bucket}/{key}"

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        if self.mode != "hmac":
            raise RuntimeError(
                "Presigned URLs require HMAC credentials; IAM mode does not support presign."
            )
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.settings.cos_bucket, "Key": key},
            ExpiresIn=expires_in,
        )
