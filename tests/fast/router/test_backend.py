from argparse import Namespace

import pytest

from miles.router.backend import _BACKEND_CLASSES, RouterKind, make_router_backend


def test_every_router_kind_has_its_own_backend():
    """A kind without a backend would only fail once a run selects it."""
    assert set(_BACKEND_CLASSES) == set(RouterKind)
    assert all(backend.kind is kind for kind, backend in _BACKEND_CLASSES.items())


@pytest.mark.parametrize(
    "router, policy, expected",
    [
        ("sglang", "consistent_hashing", {"X-SMG-Routing-Key": "key-1"}),
        ("sglang", "manual", {"X-SMG-Routing-Key": "key-1"}),
        ("sglang", "cache_aware", {}),
        ("sglang", None, {}),
        ("miles", "manual", {}),
    ],
    ids=[
        "sgl-router-consistent-hashing",
        "sgl-router-manual",
        "sgl-router-cache-aware",
        "sgl-router-default",
        "miles-router",
    ],
)
def test_a_routing_key_header_is_sent_only_to_a_router_that_reads_it(router, policy, expected):
    backend = make_router_backend(Namespace(router=router, sglang_router_policy=policy))

    assert backend.routing_headers("key-1") == expected


def test_a_key_routed_request_without_a_key_is_rejected():
    """Without the header a key policy falls back to an implicit key and can pin every request to one engine."""
    backend = make_router_backend(Namespace(router="sglang", sglang_router_policy="manual"))

    with pytest.raises(ValueError, match="routes by routing key"):
        backend.routing_headers(None)
