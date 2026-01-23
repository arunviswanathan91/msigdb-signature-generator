"""
Groq Model Diagnostic Tool
===========================

Tests Groq API connectivity and lists available models.
Use this before running debates to ensure models exist.
"""

import asyncio
from openai import AsyncOpenAI
import sys


async def diagnose_groq_models(api_key: str) -> dict:
    """
    Diagnose Groq API availability and list models.

    Args:
        api_key: Groq API key

    Returns:
        dict with 'available_models', 'test_results', 'errors'
    """
    base_url = "https://api.groq.com/openai/v1"
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    result = {
        'available_models': [],
        'test_results': {},
        'errors': [],
        'recommended_models': {}
    }

    # Step 1: List available models
    try:
        print("🔍 Fetching available Groq models...")
        models_response = await client.models.list()
        result['available_models'] = [model.id for model in models_response.data]
        print(f"✅ Found {len(result['available_models'])} models")

        for model_id in result['available_models']:
            print(f"   • {model_id}")

    except Exception as e:
        error_msg = f"Failed to list models: {str(e)}"
        result['errors'].append(error_msg)
        print(f"❌ {error_msg}")
        return result

    # Step 2: Test specific models used in debate system
    debate_models = {
        "skeptic": "llama-3.3-70b-versatile",
        "discoverer": "llama-3.1-70b-versatile",
        "mediator": "llama-3.2-90b-text-preview"
    }

    print("\n🧪 Testing debate system models...")

    for role, model_id in debate_models.items():
        test_prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'OK' if you can read this."}
        ]

        try:
            print(f"   Testing {role} ({model_id})...", end=" ")

            response = await client.chat.completions.create(
                model=model_id,
                messages=test_prompt,
                max_tokens=10,
                timeout=10.0
            )

            content = response.choices[0].message.content
            result['test_results'][role] = {
                'model_id': model_id,
                'status': 'success',
                'response': content
            }
            print(f"✅ OK")

        except Exception as e:
            error_detail = str(e)
            result['test_results'][role] = {
                'model_id': model_id,
                'status': 'failed',
                'error': error_detail
            }
            print(f"❌ FAILED")
            print(f"      Error: {error_detail}")

            # Suggest alternative if model not found
            if "model_not_found" in error_detail.lower() or "does not exist" in error_detail.lower():
                alternatives = find_similar_models(model_id, result['available_models'])
                if alternatives:
                    print(f"      💡 Try instead: {alternatives[0]}")

    # Step 3: Recommend working models for debate
    print("\n🎯 Recommended model configuration:")

    working_models = {
        role: data['model_id']
        for role, data in result['test_results'].items()
        if data['status'] == 'success'
    }

    # Fill in missing roles with alternatives
    if len(working_models) < 3:
        print("   ⚠️  Not all models working. Suggesting alternatives...")

        # Find 3 diverse models from available list
        recommended = suggest_debate_models(result['available_models'])
        result['recommended_models'] = recommended

        for role, model_id in recommended.items():
            print(f"   {role}: {model_id}")
    else:
        result['recommended_models'] = working_models
        for role, model_id in working_models.items():
            print(f"   ✅ {role}: {model_id}")

    return result


def find_similar_models(target: str, available: list) -> list:
    """Find models with similar names."""
    target_lower = target.lower()

    # Extract key parts
    if 'gemma' in target_lower:
        candidates = [m for m in available if 'gemma' in m.lower()]
    elif 'llama' in target_lower:
        candidates = [m for m in available if 'llama' in m.lower()]
    elif 'phi' in target_lower:
        candidates = [m for m in available if 'phi' in m.lower()]
    else:
        candidates = available

    return sorted(candidates)[:3]


def suggest_debate_models(available_models: list) -> dict:
    """
    Suggest 3 diverse models for debate from available list.

    Prefers:
    1. One large model (70B)
    2. One medium model (8B-13B)
    3. One alternative architecture
    """
    suggestions = {
        'qwen': None,
        'zephyr': None,
        'phi': None
    }

    # Prioritize by size
    large_models = [m for m in available_models if '70b' in m.lower() or '72b' in m.lower()]
    medium_models = [m for m in available_models if any(x in m.lower() for x in ['8b', '9b', '13b'])]
    small_models = [m for m in available_models if any(x in m.lower() for x in ['3b', '7b'])]

    # Assign roles
    if large_models:
        suggestions['qwen'] = large_models[0]

    if medium_models:
        suggestions['zephyr'] = medium_models[0]
        if len(medium_models) > 1:
            suggestions['phi'] = medium_models[1]

    # Fallback: use any available
    if not suggestions['phi'] and small_models:
        suggestions['phi'] = small_models[0]

    # If still missing, duplicate
    if not suggestions['phi']:
        suggestions['phi'] = suggestions['zephyr']

    return suggestions


async def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python groq_model_diagnostic.py <GROQ_API_KEY>")
        print("\nExample:")
        print("  python groq_model_diagnostic.py gsk_xxx...")
        sys.exit(1)

    api_key = sys.argv[1]

    print("=" * 60)
    print("Groq Model Diagnostic Tool")
    print("=" * 60)
    print()

    result = await diagnose_groq_models(api_key)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if result['errors']:
        print("❌ Errors encountered:")
        for err in result['errors']:
            print(f"   • {err}")
    else:
        print(f"✅ API connection successful")
        print(f"✅ Found {len(result['available_models'])} available models")

        working = sum(1 for r in result['test_results'].values() if r['status'] == 'success')
        total = len(result['test_results'])
        print(f"✅ {working}/{total} debate models working")

    print("\n💡 Copy this configuration to your debate_system_with_injector.py:")
    print("\n```python")
    print("self.models = {")
    for role, model_id in result['recommended_models'].items():
        print(f'    "{role}": "{model_id}",')
    print("}")
    print("```")


if __name__ == "__main__":
    asyncio.run(main())
