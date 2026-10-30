import io

from PIL import Image

from sixdegrees import search
from sixdegrees.search import CommonsSource, Hit, Person, collect_images, download, resolve_person
from sixdegrees.util import list_images, read_json


def test_resolve_person_skips_disambiguation(monkeypatch):
    calls = []

    def fake_get(url, params, namespace, ttl_days=30):
        calls.append(params)
        if params.get("list") == "search":
            return {"query": {"search": [{"title": "Diddy (disambiguation)"}, {"title": "Sean Combs"}]}}
        title = params["titles"]
        if title == "Diddy":
            return {"query": {"pages": [{"title": "Diddy", "pageprops": {"disambiguation": ""}}]}}
        if title == "Diddy (disambiguation)":
            return {"query": {"pages": [{"title": title, "pageprops": {"disambiguation": ""}}]}}
        return {"query": {"pages": [{"title": "Sean Combs", "pageprops": {"wikibase_item": "Q216936"},
                                     "thumbnail": {"source": "https://upload.example/combs.jpg"},
                                     "description": "American rapper"}]}}

    monkeypatch.setattr(search, "cached_get_json", fake_get)
    p = resolve_person("Diddy")
    assert (p.name, p.title, p.qid, p.image_url) == ("Diddy", "Sean Combs", "Q216936", "https://upload.example/combs.jpg")


def test_resolve_person_falls_back_when_offline(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("429")

    monkeypatch.setattr(search, "cached_get_json", boom)
    assert resolve_person("Someone").title == "Someone"


def test_commons_search_uses_depicts_then_names(monkeypatch):
    queries = []

    def fake_get(url, params, namespace, ttl_days=30):
        queries.append(params["gsrsearch"])
        return {"query": {"pages": []}}

    monkeypatch.setattr(search, "cached_get_json", fake_get)
    people = [Person("a", "Barack Obama", "Q76"), Person("b", "Pope Francis", "Q450675")]
    CommonsSource().search(people, "Obama Pope Francis White House 2015", 10)
    assert queries == ["haswbstatement:P180=Q76 haswbstatement:P180=Q450675",
                       '"Barack Obama" "Pope Francis" filetype:bitmap',
                       "Obama Pope Francis White House 2015 filetype:bitmap"]


def _png_bytes(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        yield self.data


class FakeSession:
    def __init__(self, data_by_url):
        self.data_by_url = data_by_url

    def get(self, url, **kw):
        return FakeResponse(self.data_by_url[url])


def test_download_saves_valid_images_and_records_source(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "http", lambda: FakeSession({"u/big": _png_bytes(400, 300), "u/tiny": _png_bytes(50, 50)}))
    got = download(Hit("u/big", "commons", "File:x.png", "page"), tmp_path)
    assert got is not None and got.suffix == ".png"
    assert download(Hit("u/big", "commons"), tmp_path) == got  # already there
    assert download(Hit("u/tiny", "commons"), tmp_path) is None
    assert read_json(tmp_path / "sources.json")[got.name]["url"] == "u/big"


class FakeSource:
    name = "fake"

    def __init__(self, n):
        self.n, self.calls = n, 0

    def search(self, people, query, limit):
        self.calls += 1
        return [Hit(f"u/{self.calls}/{i}", "fake") for i in range(self.n)]


def test_collect_images_stops_at_limit_and_reuses_existing(tmp_path, monkeypatch):
    urls = {f"u/{c}/{i}": _png_bytes(300, 300) for c in range(1, 4) for i in range(10)}
    monkeypatch.setattr(search, "http", lambda: FakeSession(urls))
    src = FakeSource(10)
    people = [Person("a", "A"), Person("b", "B")]
    files = collect_images(people, "A B party", tmp_path, [src], limit=5)
    assert len(files) >= 5 and src.calls == 1
    again = FakeSource(10)
    assert collect_images(people, None, tmp_path, [again], limit=5) == list_images(tmp_path)
    assert again.calls == 0
