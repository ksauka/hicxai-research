#!/bin/bash
# Quick test script for LLM validation
# Tests that validation messages are generated correctly

cd "$(dirname "$0")"

echo "🧪 Testing LLM Validation Messages..."
echo ""

python3 -c "
import sys
sys.path.insert(0, 'src')
import os

# Load environment
with open('.env', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

from natural_conversation import enhance_validation_message

# Test cases
test_cases = [
    ('age', 'gg', 'a number like 25'),
    ('income', 'abc', 'a number like 50000'),
    ('education', '123', 'one of: Bachelors, Masters, Doctorate, etc.'),
]

print('Testing LLM validation messages...\n')
success_count = 0

for field, invalid_input, expected_format in test_cases:
    result = enhance_validation_message(field, invalid_input, expected_format, attempt=1)
    
    if result:
        print(f'✅ {field}: {result[:80]}...')
        success_count += 1
    else:
        print(f'❌ {field}: No LLM response (fallback will be used)')

print(f'\n📊 Results: {success_count}/{len(test_cases)} successful')

if success_count == len(test_cases):
    print('✅ All tests passed! LLM validation is ready for deployment.')
    sys.exit(0)
else:
    print('⚠️  Some tests failed. Check your OPENAI_API_KEY and billing.')
    sys.exit(1)
"
