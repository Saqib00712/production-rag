import numpy as np
import pytest

from app.services.vector_index_cache import TenantIndexCache


def make_cache(backend="brute_force"):
    return TenantIndexCache(backend=backend, ef_construction=50, m=8, ef_search=20)


async def loader_factory(ids, matrix, calls):
    async def loader():
        calls.append(1)
        return ids, matrix

    return loader


@pytest.mark.asyncio
async def test_builds_once_and_reuses_across_calls():
    cache = make_cache()
    calls = []
    loader = await loader_factory(["a"], np.array([[1.0, 0.0]], dtype=np.float32), calls)
    await cache.get("t1", "m", loader)
    await cache.get("t1", "m", loader)
    assert len(calls) == 1  # second call reused the cached index, no rebuild


@pytest.mark.asyncio
async def test_invalidate_forces_a_rebuild_on_next_access():
    cache = make_cache()
    calls = []
    loader = await loader_factory(["a"], np.array([[1.0, 0.0]], dtype=np.float32), calls)
    await cache.get("t1", "m", loader)
    cache.invalidate("t1")
    await cache.get("t1", "m", loader)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_different_tenants_are_cached_independently():
    cache = make_cache()
    calls_a, calls_b = [], []
    loader_a = await loader_factory(["a"], np.array([[1.0, 0.0]], dtype=np.float32), calls_a)
    loader_b = await loader_factory(["b"], np.array([[0.0, 1.0]], dtype=np.float32), calls_b)
    await cache.get("tenant-a", "m", loader_a)
    await cache.get("tenant-b", "m", loader_b)
    cache.invalidate("tenant-a")
    await cache.get("tenant-a", "m", loader_a)
    await cache.get("tenant-b", "m", loader_b)  # untouched by tenant-a's invalidation
    assert len(calls_a) == 2 and len(calls_b) == 1


@pytest.mark.asyncio
async def test_different_models_for_the_same_tenant_are_cached_independently():
    cache = make_cache()
    calls_m1, calls_m2 = [], []
    loader_m1 = await loader_factory(["a"], np.array([[1.0, 0.0]], dtype=np.float32), calls_m1)
    loader_m2 = await loader_factory(["a"], np.array([[1.0, 0.0]], dtype=np.float32), calls_m2)
    await cache.get("t1", "model-1", loader_m1)
    await cache.get("t1", "model-2", loader_m2)
    assert len(calls_m1) == 1 and len(calls_m2) == 1


@pytest.mark.asyncio
async def test_reset_clears_everything():
    cache = make_cache()
    calls = []
    loader = await loader_factory(["a"], np.array([[1.0, 0.0]], dtype=np.float32), calls)
    await cache.get("t1", "m", loader)
    cache.reset()
    await cache.get("t1", "m", loader)
    assert len(calls) == 2
