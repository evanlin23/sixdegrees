from sixdegrees.pipeline import Options, Pipeline
from sixdegrees.planners import Link, links_from_people
from sixdegrees.search import Person, make_sources, parse_commons_response
from sixdegrees.util import read_json


class ScriptedPlanner:
    def __init__(self, plan, repairs):
        self._plan, self._repairs, self.calls = plan, list(repairs), []

    def plan(self, start, target):
        return links_from_people(self._plan)

    def repair(self, verified, target, failed, remaining):
        self.calls.append((list(verified), [(l.a, l.b, r) for l, r in failed]))
        nxt = self._repairs.pop(0) if self._repairs else None
        return links_from_people(nxt) if nxt else None


def make_pipeline(tmp_path, photos):
    """photos: set of (a, b) pairs that "have a photo"."""
    pipe = Pipeline(Options(out_dir=tmp_path, max_repairs=3))
    pipe.person = lambda name: Person(name, name)

    def verify(link, run_dir):
        ok = (link.a, link.b) in photos
        return {"a": link.a, "b": link.b, "status": "ok" if ok else "failed",
                "reason": "" if ok else "no photo", "image": "x.jpg" if ok else None}

    pipe.verify_link = verify
    return pipe


def test_run_repairs_around_failed_link(tmp_path):
    pipe = make_pipeline(tmp_path, {("A", "B"), ("B", "D"), ("D", "C")})
    planner = ScriptedPlanner(["A", "B", "C"], [["B", "D", "C"]])
    path = pipe.run("A", "C", planner)
    state = read_json(path)
    assert state["complete"] and state["people"] == ["A", "B", "D", "C"]
    assert planner.calls == [(["A", "B"], [("B", "C", "no photo")])]
    assert [f["b"] for f in state["failed"]] == ["C"]


def test_run_stops_when_planner_gives_up(tmp_path):
    pipe = make_pipeline(tmp_path, {("A", "B")})
    state = read_json(pipe.run("A", "C", ScriptedPlanner(["A", "B", "C"], [])))
    assert not state["complete"] and state["people"] == ["A", "B"]
    assert "no alternative" in state["message"]


def test_run_rejects_loops_and_respects_max_repairs(tmp_path):
    pipe = make_pipeline(tmp_path, {("A", "B"), ("B", "A")})
    planner = ScriptedPlanner(["A", "B", "A", "C"], [["B", "A", "C"]] * 5)
    state = read_json(pipe.run("A", "C", planner))
    assert not state["complete"] and len(planner.calls) == 3
    assert "stopped after 3 repairs" in state["message"]


def test_parse_commons_response_filters_and_orders():
    data = {"query": {"pages": [
        {"title": "File:b.jpg", "index": 2, "imageinfo": [{"url": "u2", "thumburl": "t2", "width": 2000, "height": 1500,
                                                           "mime": "image/jpeg", "descriptionurl": "d2"}]},
        {"title": "File:a.jpg", "index": 1, "imageinfo": [{"url": "u1", "width": 800, "height": 600,
                                                           "mime": "image/png", "descriptionurl": "d1"}]},
        {"title": "File:tiny.jpg", "index": 3, "imageinfo": [{"url": "u3", "width": 100, "height": 80,
                                                              "mime": "image/jpeg"}]},
        {"title": "File:doc.pdf", "index": 4, "imageinfo": [{"url": "u4", "width": 900, "height": 900,
                                                             "mime": "application/pdf"}]},
    ]}}
    hits = parse_commons_response(data)
    assert [h.url for h in hits] == ["u1", "t2"]
    assert hits[0].page == "d1"


def test_make_sources(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "k")
    assert make_sources("none") == []
    assert [s.name for s in make_sources("commons")] == ["commons"]
    assert [s.name for s in make_sources("commons,brave,serpapi")] == ["commons", "serpapi"]
