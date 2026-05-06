import re


def validate_target_date(target_date: str) -> None:
    if not re.fullmatch(r"\d{8}", target_date):
        raise ValueError("target_date must be YYYYMMDD")
