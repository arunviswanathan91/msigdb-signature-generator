# Quick Start: Testing Debate System Fixes

## 🚀 1-Minute Test

Run this to verify everything works:

```bash
# Step 1: Test your Groq models
python groq_model_diagnostic.py gsk_your_api_key_here

# Step 2: If models need updating, the tool will tell you exactly what to change
```

## 📝 What the Diagnostic Tool Does

1. **Connects to Groq API** and lists all available models
2. **Tests each debate model** (qwen, zephyr, phi) to verify they work
3. **Suggests alternatives** if any model is unavailable
4. **Provides ready-to-use configuration** to copy into your code

## Example Output

```
🔍 Fetching available Groq models...
✅ Found 12 models
   • llama-3.3-70b-versatile
   • llama-3.1-8b-instant
   • gemma-7b-it
   ...

🧪 Testing debate system models...
   Testing qwen (llama-3.3-70b-versatile)... ✅ OK
   Testing zephyr (llama-3.1-8b-instant)... ✅ OK
   Testing phi (gemma2-9b-it)... ❌ FAILED
      Error: model not found
      💡 Try instead: gemma-7b-it

💡 Copy this configuration to your debate_system_with_injector.py:

```python
self.models = {
    "qwen": "llama-3.3-70b-versatile",
    "zephyr": "llama-3.1-8b-instant",
    "phi": "gemma-7b-it",  # ← Fixed!
}
```
```

## 🔧 Applying the Fix

If you see "FAILED" for any model:

1. **Open** `debate_system_with_injector.py`
2. **Find** lines 165-169 (the `self.models = {...}` section)
3. **Replace** the broken model ID with the suggested one
4. **Save** the file
5. **Restart** your Streamlit app

## ✅ That's It!

The debate system will now:
- ✅ Properly detect model errors
- ✅ Display errors clearly in the UI
- ✅ Validate models on startup (optional)
- ✅ Support JSON mode for better accuracy (optional)

## 🆘 Still Having Issues?

Read the full documentation: `DEBATE_SYSTEM_FIXES.md`

Or test with this minimal Python script:

```python
import asyncio
from debate_system_with_injector import MultiRoundDebateEngine
from db_client_enhanced import DatabaseClientEnhanced

async def test():
    db = DatabaseClientEnhanced("https://arunviswanathan91-msigdb-api.hf.space")

    engine = MultiRoundDebateEngine(
        api_key="gsk_your_api_key",
        db_client=db,
        validate_models=True  # Will print validation results
    )

    result = await engine.run_validation_debate(
        genes=["TP53", "EGFR"],
        max_rounds=1
    )

    print(f"✅ Debate completed!")
    print(f"Decision: {result.final_decision}")
    print(f"Confidence: {result.confidence:.2%}")

asyncio.run(test())
```

Expected output:
```
🔍 Validating Groq models...
   ✅ qwen: llama-3.3-70b-versatile
   ✅ zephyr: llama-3.1-8b-instant
   ✅ phi: gemma-7b-it
✅ Debate completed!
Decision: keep
Confidence: 75.00%
```
