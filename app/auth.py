from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException, Depends, status
from config import Settings, get_settings

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def get_api_key(
    settings: Settings = Depends(get_settings),
    api_key_header: str = Security(api_key_header),
):
    if api_key_header == settings.API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
