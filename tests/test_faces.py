from pathlib import Path

import numpy as np
import pytest

from sixdegrees.faces import Face, Identity, PairMatch, match_pair

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "old_tries" / "full" / "reference_faces"


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def face_with(emb, x=0.0, size=100.0):
    return Face(x, 0, size, size, 1.0, [], unit(emb))


def test_match_pair_assigns_distinct_faces():
    e1, e2 = np.eye(128)[0], np.eye(128)[1]
    ida, idb = Identity("A", np.stack([unit(e1)])), Identity("B", np.stack([unit(e2)]))
    f_a, f_b, f_x = face_with(e1, 0), face_with(e2, 200), face_with(np.eye(128)[2], 400)
    m = match_pair([f_x, f_b, f_a], ida, idb)
    assert m.a is f_a and m.b is f_b and m.ok()
    # only A present: B gets the stranger, and the match is rejected
    m = match_pair([f_a, f_x], ida, idb)
    assert m.b is f_x and not m.ok()
    assert match_pair([f_a], ida, idb) is None


def test_pair_score_prefers_bigger_faces_and_fewer_people():
    big = PairMatch(face_with(np.ones(128), size=200), face_with(np.ones(128), size=200), 0.6, 0.6, 2)
    small = PairMatch(face_with(np.ones(128), size=40), face_with(np.ones(128), size=40), 0.6, 0.6, 2)
    crowd = PairMatch(big.a, big.b, 0.6, 0.6, 12)
    assert big.score > small.score and big.score > crowd.score


def _engine():
    try:
        from sixdegrees.faces import FaceEngine

        return FaceEngine()
    except Exception as e:  # models not downloadable here
        pytest.skip(f"face models unavailable: {e}")


@pytest.mark.skipif(not REFS.exists(), reason="sample photos not present")
def test_real_photo_identifies_both_people():
    from sixdegrees.faces import build_identity, looks_like_collage
    from sixdegrees.util import list_images, load_image

    eng = _engine()
    together = REFS / "Portia_de_Rossi" / "000005.jpg"  # Ellen and Portia side by side
    ellen = build_identity("Ellen", [p for p in list_images(REFS / "Ellen_DeGeneres")], eng)
    portia = build_identity("Portia", [p for p in list_images(REFS / "Portia_de_Rossi") if p != together], eng)
    m = match_pair(eng.faces_in(together), ellen, portia)
    assert m.ok(), (m.sim_a, m.sim_b)
    assert m.a.center[0] < m.b.center[0]  # Ellen is on the left
    assert not looks_like_collage(load_image(together), m.a, m.b)
    # A side-by-side composite of two portraits is caught
    trump_vance = REFS / "Donald_Trump" / "000002.jpg"
    faces = eng.faces_in(trump_vance)
    assert looks_like_collage(load_image(trump_vance), faces[0], faces[1])
