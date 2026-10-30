import pytest

from sixdegrees.planners import (GeminiPlanner, GraphPlanner, Link, ManualPlanner, PlannerError, extract_json,
                                 normalize_chain)


def test_extract_json_handles_fences_chatter_and_trailing_commas():
    assert extract_json('```json\n{"chain": []}\n```') == {"chain": []}
    assert extract_json('Sure! Here you go: {"a": 1, "b": [1, 2,],} Hope that helps') == {"a": 1, "b": [1, 2]}
    assert extract_json("no json here") is None
    assert extract_json("") is None


def chain(*pairs):
    return {"chain": [{"from": a, "to": b, "occasion": "o", "year": "2000", "query": "q"} for a, b in pairs]}


def test_normalize_chain_snaps_names_and_validates_continuity():
    links = normalize_chain(chain(("Donald J. Trump", "Oprah Winfrey"), ("Oprah", "Sean Combs")),
                            "Donald Trump", "Sean Combs")
    assert [(l.a, l.b) for l in links] == [("Donald Trump", "Oprah Winfrey"), ("Oprah Winfrey", "Sean Combs")]
    # broken chain (second link doesn't start where the first ended)
    assert normalize_chain(chain(("A Person", "B Person"), ("C Person", "D Person")), "A Person", "D Person") is None
    # wrong endpoints
    assert normalize_chain(chain(("Xavier Q", "Yolanda Z")), "A Person", "B Person") is None
    # repeated person
    assert normalize_chain(chain(("Ann Lee", "Bob Ray"), ("Bob Ray", "Ann Lee"), ("Ann Lee", "Cat Moe")),
                           "Ann Lee", "Cat Moe") is None
    assert normalize_chain({"chain": []}, "a", "b") is None


def test_manual_planner():
    p = ManualPlanner(["A", "B", "C"])
    assert [(l.a, l.b) for l in p.plan("A", "C")] == [("A", "B"), ("B", "C")]
    assert p.repair(["A"], "C", [(Link("A", "B"), "x")], []) is None
    with pytest.raises(PlannerError):
        ManualPlanner(["A"])


GEXF = """<?xml version='1.0' encoding='utf-8'?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2"><graph defaultedgetype="undirected">
<nodes><node id="1" label="Ann Lee"/><node id="2" label="Bob Ray"/><node id="3" label="Cat Moe"/>
<node id="4" label="Dan Fox"/><node id="5" label="Eve Kim"/></nodes>
<edges><edge source="1" target="2"/><edge source="2" target="3"/><edge source="1" target="4"/>
<edge source="4" target="5"/><edge source="5" target="3"/></edges></graph></gexf>"""


def test_graph_planner_plans_and_routes_around_failures(tmp_path):
    path = tmp_path / "g.gexf"
    path.write_text(GEXF)
    g = GraphPlanner(path)
    links = g.plan("ann lee", "Cat Moe")
    assert [l.b for l in links] == ["Bob Ray", "Cat Moe"]
    assert links[0].a == "ann lee"  # endpoints keep the caller's spelling
    new = g.repair(["ann lee"], "Cat Moe", [(links[0], "no photo")], links[1:])
    assert [l.b for l in new] == ["Dan Fox", "Eve Kim", "Cat Moe"]
    # once Dan Fox -> Eve Kim also fails there is no route left
    assert g.repair(["ann lee", "Dan Fox"], "Cat Moe", [(Link("Dan Fox", "Eve Kim"), "x")], []) is None
    with pytest.raises(PlannerError):
        g.node("Nobody Here")


class FakeGemini(GeminiPlanner):
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.model, self.grounding, self.use_cache = "fake", True, True

    def _generate(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_gemini_planner_retries_bad_replies_and_caches():
    good = '{"chain": [{"from": "Ann Lee", "to": "Bob Ray"}, {"from": "Bob Ray", "to": "Cat Moe"}], "notes": "n"}'
    p = FakeGemini(["I think they met once.", good])
    links = p.plan("Ann Lee", "Cat Moe")
    assert [l.b for l in links] == ["Bob Ray", "Cat Moe"]
    assert len(p.prompts) == 2 and "not usable" in p.prompts[1]
    cached = FakeGemini([])  # no replies needed: answer comes from the cache
    assert [l.b for l in cached.plan("Ann Lee", "Cat Moe")] == ["Bob Ray", "Cat Moe"]


def test_gemini_planner_drops_grounding_when_tool_unsupported():
    good = '{"chain": [{"from": "Ann Lee", "to": "Cat Moe"}]}'
    p = FakeGemini([RuntimeError("tools not supported"), good])
    assert [l.b for l in p.plan("Ann Lee", "Cat Moe")] == ["Cat Moe"]
    assert p.grounding is False


def test_gemini_repair_prompt_mentions_failures_and_used_people():
    good = '{"chain": [{"from": "Bob Ray", "to": "Dan Fox"}, {"from": "Dan Fox", "to": "Cat Moe"}]}'
    p = FakeGemini([good])
    links = p.repair(["Ann Lee", "Bob Ray"], "Cat Moe", [(Link("Bob Ray", "Cat Moe", "Gala", "2001"), "no photos")], [])
    assert [(l.a, l.b) for l in links] == [("Bob Ray", "Dan Fox"), ("Dan Fox", "Cat Moe")]
    assert "Bob Ray & Cat Moe (Gala, 2001): no photos" in p.prompts[0]
    assert "Ann Lee" in p.prompts[0]
