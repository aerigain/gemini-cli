import { BedrockContentGenerator } from './packages/core/src/core/providers/bedrockProvider.js';
import { LlmRole } from './packages/core/src/telemetry/llmRole.js';

async function main() {
  const generator = new BedrockContentGenerator('us-east-1');
  const request = {
    model: 'us.amazon.nova-lite-v1:0',
    contents: [{ role: 'user', parts: [{ text: 'Hello, please respond with "Bedrock is working!"' }] }],
  };

  console.log('Testing Bedrock Nova 2 Lite...');
  try {
    const response = await generator.generateContent(request as any, 'test-id', LlmRole.MAIN);
    console.log('Response:', response.candidates?.[0].content?.parts?.[0].text);
    if (response.candidates?.[0].content?.parts?.[0].text?.includes('Bedrock is working')) {
      console.log('SUCCESS: Bedrock integration validated.');
    } else {
      console.log('FAILURE: Unexpected response.');
    }
  } catch (error) {
    console.error('ERROR:', error);
  }
}

main();
