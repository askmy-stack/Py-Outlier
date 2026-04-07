from pydantic_settings import BaseSettings, SettingsConfigDict


CRIME_CLASSES = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "Normal",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Anonmaly Detection API"
    debug: bool = False
    version: str = "1.0.0"

    # Model paths (relative to project root or absolute)
    model_image_path: str = "../Image Anomaly Detection-2"
    model_video_path: str = "../Video Anomaly Detection"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # LLM
    anthropic_api_key: str = ""

    # Observability
    sentry_dsn: str = ""

    # Upload limits
    max_upload_size_mb: int = 50

    # CORS
    allowed_origins: list[str] = ["http://localhost:3000"]

    # ML class labels
    classes: list[str] = CRIME_CLASSES

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
