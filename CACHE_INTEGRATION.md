# Search Cache Integration

## Overview

The Streamlit app now uses a remote search cache to speed up repeated queries.

## How It Works

1. **First Query**: When you search for a mechanism (e.g., "IL-6 signaling"), the app:
   - Computes semantic similarity with all pathways (~5 seconds)
   - Stores results in the remote cache
   - Returns results

2. **Repeat Query**: When you search for the same mechanism again, the app:
   - Checks the cache first (~0.5 seconds)
   - Returns cached results immediately
   - Skips computation entirely

## Cache Location

- **Remote API**: https://arunviswanathan91-msigdb-api.hf.space
- **Endpoints**:
  - `POST /api/cache/search` - Query cache
  - `POST /api/cache/store` - Store results
  - `GET /api/cache/stats` - Get statistics

## Files Modified

### New Files
- `cache_client.py` - Cache client implementation
- `test_cache_integration.py` - Test suite
- `CACHE_INTEGRATION.md` - This documentation

### Modified Files
- `app.py`:
  - Added cache client import
  - Modified `render_layer2_semantic_fixed()` to use cache
  - Added cache stats to sidebar
- `requirements.txt`:
  - Added `requests>=2.31.0`

## Usage

The cache is **automatic** - no user action required.

### Monitoring Cache Performance

Check the sidebar "📊 Search Cache Stats" section to see:
- Total cached searches
- Cache hit count
- Average reuse rate

### Cache Hit Display

During signature generation (Layer 2), you'll see:
```
📊 Cache Performance: 5/10 hits (50.0%)
```

This means 5 out of 10 searches used the cache.

## Performance Impact

| Scenario | Before Cache | With Cache (50% hits) | With Cache (90% hits) |
|----------|--------------|----------------------|---------------------|
| Layer 2 Time | 45-60s | 25-35s | 10-15s |
| API Calls | 10-30 | 5-15 | 1-3 |

## Testing

Run the test suite:

```bash
python test_cache_integration.py
```

Expected output:
```
======================================================================
TESTING SEARCH CACHE CLIENT
======================================================================
...
✅ ALL TESTS PASSED
======================================================================
```

## Troubleshooting

### "Cache unavailable" in sidebar
- **Cause**: HuggingFace Space is sleeping
- **Fix**: Visit https://arunviswanathan91-msigdb-api.hf.space to wake it up

### "Cache Performance: 0/10 hits"
- **Cause**: All queries are new (first time running)
- **Fix**: Normal behavior - cache will populate over time

### Cache always misses
- **Cause**: KB version mismatch
- **Fix**: Check `cache_client.py` - ensure `kb_version = "2025.1"` matches API

## Cache Invalidation

When the knowledge base is updated:

1. Update `kb_version` in `cache_client.py`:
   ```python
   self.kb_version = "2025.2"  # New version
   ```

2. Old cache entries will be ignored automatically

## Privacy

The cache stores:
- ✅ Query hashes (anonymized)
- ✅ Pathway IDs and similarity scores
- ❌ NO user identifiers
- ❌ NO IP addresses
- ❌ NO session data

All cached data is anonymous and shared across users.

## Future Enhancements

Potential improvements:
- [ ] Pre-populate cache with common queries
- [ ] Add cache warming on app startup
- [ ] Display cache age/freshness
- [ ] Add manual cache invalidation button
