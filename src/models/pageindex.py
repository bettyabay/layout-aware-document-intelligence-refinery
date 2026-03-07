from __future__ import annotations

from pydantic import BaseModel, Field


class PageIndexSection(BaseModel):
    section_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    summary: str = ""
    key_entities: list[str] = Field(default_factory=list)
    data_types_present: list[str] = Field(default_factory=list)
    child_sections: list["PageIndexSection"] = Field(default_factory=list)


class PageIndex(BaseModel):
    doc_id: str = Field(min_length=1)
    root: PageIndexSection

    def _all_sections(self, section: PageIndexSection | None = None) -> list[PageIndexSection]:
        section = section or self.root
        out: list[PageIndexSection] = [section]
        for child in section.child_sections or []:
            out.extend(self._all_sections(child))
        return out

    def top_sections_for_topic(self, topic: str, k: int = 3) -> list[PageIndexSection]:
        topic_tokens = set((topic or "").lower().split())
        candidates = self._all_sections(self.root)[1:]

        def score(section: PageIndexSection) -> int:
            title_tokens = set(section.title.lower().split())
            summary_tokens = set((section.summary or "").lower().split())
            entity_tokens = set()
            for e in section.key_entities or []:
                entity_tokens.update(e.lower().split())
            combined = title_tokens | summary_tokens | entity_tokens
            return len(topic_tokens.intersection(combined))

        ranked = sorted(candidates, key=lambda s: (-score(s), s.page_start))
        return ranked[:k]
