from typing import Optional, Dict, Any
from datetime import datetime
from sqlmodel import SQLModel, Field
import json

class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    created_utc: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    input_type: str
    input_url: Optional[str] = None
    input_domain: Optional[str] = None

    extracted_text_chars: int = 0
    extracted_text_preview: str = ""

    result_json: str

    def set_result(self, obj: Dict[str, Any]) -> None:
        self.result_json = json.dumps(obj, ensure_ascii=False)

    def get_result(self) -> Dict[str, Any]:
        return json.loads(self.result_json)
